import re
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

DATA_PATH = Path("data/pdp_sample.csv")

st.set_page_config(page_title="IntentifAI × STIGA – PDP Chat (MVP)", page_icon="🟡", layout="wide")
st.title("IntentifAI × STIGA – PDP Chat (MVP)")
st.caption("Demo interna: risposte basate su contenuti PDP indicizzati (con citazioni URL).")

# ---------- Utilities ----------
def build_passage(row: pd.Series) -> str:
    parts = [
        f"Title: {row.get('title','')}",
        f"Category: {row.get('category','')}",
        f"Coverage (m2): {row.get('coverage_m2','')}",
        f"Cut height (mm): {row.get('cutting_height_min_mm','')}–{row.get('cutting_height_max_mm','')}",
        f"Noise (dB): {row.get('noise_db','')}",
        f"Slope max (%): {row.get('slope_max_percent','')}",
        f"Compatibility: {row.get('compatibility','')}",
        f"Features: {row.get('features','')}",
        f"Price (EUR): {row.get('price_eur','')}",
        f"PDP: {row.get('pdp_url','')}",
    ]
    return "\n".join([p for p in parts if p and str(p).strip()])

def extract_numbers(text: str) -> Dict[str, float]:
    """Heuristics: estrai m², dB, pendenza % se presenti nella query per un ranking più intelligente."""
    t = text.lower().replace(",", ".")
    res = {}
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*(m2|mq|m²)", t)
    if m2: res["coverage_m2"] = float(m2.group(1))
    db = re.search(r"(\d+(?:\.\d+)?)\s*dB", t)
    if db: res["noise_db"] = float(db.group(1))
    slope = re.search(r"(\d+(?:\.\d+)?)\s*%(\s*di)?\s*(pendenza|slope)?", t)
    if slope: res["slope_max_percent"] = float(slope.group(1))
    return res

@st.cache_resource(show_spinner=True)
def load_data_and_index() -> Tuple[pd.DataFrame, List[str], List[Dict[str,Any]], Any, Any]:
    if not DATA_PATH.exists():
        st.error("Manca data/pdp_sample.csv. Caricalo nel repo.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    # Passaggi testuali + meta
    passages = []
    meta = []
    for _, r in df.iterrows():
        passages.append(build_passage(r))
        meta.append({
            "sku": r.get("sku",""),
            "title": r.get("title",""),
            "pdp_url": r.get("pdp_url",""),
            "category": r.get("category",""),
            "coverage_m2": r.get("coverage_m2", np.nan),
            "cut_min": r.get("cutting_height_min_mm", np.nan),
            "cut_max": r.get("cutting_height_max_mm", np.nan),
            "noise_db": r.get("noise_db", np.nan),
            "slope_max_percent": r.get("slope_max_percent", np.nan),
            "price_eur": r.get("price_eur", np.nan),
        })
    # Embeddings + index NN (cosine)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(passages, normalize_embeddings=True)
    nn = NearestNeighbors(n_neighbors=8, metric="cosine")
    nn.fit(emb)
    return df, passages, meta, model, nn

def retrieve(query: str, model, nn, passages, meta, top_k=6):
    q_emb = model.encode([query], normalize_embeddings=True)
    dist, idx = nn.kneighbors(q_emb, n_neighbors=top_k)
    hits = []
    for d, i in zip(dist[0], idx[0]):
        hits.append({"score": float(1 - d), "passage": passages[i], "meta": meta[i]})
    return hits

def score_fit(h: Dict[str,Any], wish: Dict[str,float]) -> float:
    """Rerank semplice usando vincoli utente (m2, dB, pendenza)."""
    fit = 0.0
    m = h["meta"]
    # Copertura: preferisci coverage >= richiesta (entro +50%)
    if "coverage_m2" in wish and pd.notna(m.get("coverage_m2")):
        w = wish["coverage_m2"]; c = float(m["coverage_m2"])
        if c >= w: 
            # più vicino è al target, meglio è
            fit += 1.0 - min((c - w)/max(w,1), 0.5)
    # Rumore: preferisci <= soglia
    if "noise_db" in wish and pd.notna(m.get("noise_db")):
        w = wish["noise_db"]; n = float(m["noise_db"])
        if n <= w:
            fit += 1.0 - min((w - n)/max(w,1), 0.5)
    # Pendenza: preferisci slope_max >= richiesta
    if "slope_max_percent" in wish and pd.notna(m.get("slope_max_percent")):
        w = wish["slope_max_percent"]; s = float(m["slope_max_percent"])
        if s >= w:
            fit += 1.0 - min((s - w)/max(w,1), 0.5)
    return fit

def compose_answer(query: str, hits: List[Dict[str,Any]], wish: Dict[str,float]) -> str:
    if not hits:
        return "Non trovo info sufficienti. Puoi indicare area (m²), pendenza (%) o target di rumorosità (dB)?"
    # Rerank by semantic score + wish fit
    for h in hits:
        h["fit"] = score_fit(h, wish)
        h["tot"] = 0.7*h["score"] + 0.3*h["fit"]
    hits = sorted(hits, key=lambda x: x["tot"], reverse=True)

    lines = [f"**Domanda:** {query}"]
    lines.append("**Opzioni consigliate (ordinate per pertinenza):**")
    for h in hits:
        m = h["meta"]
        lines.append(
            f"- **{m['title']}** — copertura **{m.get('coverage_m2','?')} m²**, pendenza max **{m.get('slope_max_percent','?')}%**, "
            f"rumorosità **{m.get('noise_db','?')} dB** — [PDP]({m['pdp_url']})"
        )
    cites = " | ".join(sorted({h['meta']['pdp_url'] for h in hits if h['meta']['pdp_url']}))
    lines.append(f"\n**Fonti PDP:** {cites}")
    lines.append("\nVuoi **confrontare** due modelli, verificare **compatibilità batteria**, o preferisci quello **più silenzioso**?")
    return "\n".join(lines)

# ---------- App ----------
with st.sidebar:
    st.subheader("Filtri/indicazioni (estratti automaticamente dalla domanda)")
    st.write("Inserisci nella domanda indicazioni tipo: `600 m²`, `<60 dB`, `pendenza 35%`.")
    st.divider()
    st.write("Esempi:")
    st.code("robot per 600 m², pendenza 35%, sotto 60 dB")
    st.code("A750: altezza minima di taglio?")
    st.code("compatibilità ePower 20V 4Ah con A1000? (demo)")

df, passages, meta, model, nn = load_data_and_index()

query = st.text_input("Fai una domanda sulle PDP STIGA (IT/EN):", "")
if query:
    wish = extract_numbers(query)
    with st.spinner("Cerco tra le PDP…"):
        hits = retrieve(query, model, nn, passages, meta)
        answer = compose_answer(query, hits, wish)
    st.markdown(answer)
    with st.expander("🔎 Passaggi usati (debug)"):
        for h in hits:
            st.write(h["meta"]["title"], "—", h["meta"]["pdp_url"], f"(score={h['score']:.3f}, fit={h['fit']:.3f})")
            st.code(h["passage"])
