import re
import json
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
st.caption("Demo interna: risposte basate su contenuti PDP indicizzati (con citazioni URL) con vincoli hard-filter.")

# -------------------- Parsing vincoli --------------------
def normalize(text: str) -> str:
    return text.lower().strip()

def parse_constraints(q: str) -> Dict[str, Any]:
    """
    Estrae vincoli da query:
    - coverage (m²): '600 m²', '600 mq', '>= 800 m²', 'almeno 800 mq'
    - noise (dB): '< 60 dB', 'sotto 60 dB', '<=58 db', 'massimo 59 dB'
    - slope (%): 'pendenza 35%', '>=35%', 'almeno 35%', 'slope 35%'
    Ritorna dict con chiavi *_min / *_max.
    """
    t = normalize(q).replace(",", ".")
    c: Dict[str, Any] = {}

    # coverage m2
    cov_op_num = re.search(r"(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)\s*(m2|mq|m²)", t)
    if cov_op_num:
        op, num = cov_op_num.group(1), float(cov_op_num.group(2))
        if op in (">", ">=") or re.search(r"\b(almeno|minimo|min)\b", t):
            c["coverage_m2_min"] = num
        elif op in ("<", "<=") or re.search(r"\b(sotto|massimo|max)\b", t):
            c["coverage_m2_max"] = num
        else:
            c["coverage_m2_min"] = num

    # noise dB
    noise_phrase = re.search(r"(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)\s*d\s*b", t)
    if noise_phrase or re.search(r"(sotto|max|massimo|minimo|almeno)\s*\d+(?:\.\d+)?\s*d\s*b", t):
        if noise_phrase:
            op, num = noise_phrase.group(1), float(noise_phrase.group(2))
        else:
            m = re.search(r"(sotto|max|massimo|minimo|almeno)\s*(\d+(?:\.\d+)?)\s*d\s*b", t)
            word, num = m.group(1), float(m.group(2))
            op = "<=" if word in ("sotto", "max", "massimo") else ">="
        if op in ("<", "<="):
            c["noise_db_max"] = num
        elif op in (">", ">="):
            c["noise_db_min"] = num
        else:
            c["noise_db_max"] = num

    # slope %
    slope_any = re.search(r"(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)\s*%(\s*(pendenza|slope))?", t)
    if slope_any or re.search(r"(pendenza|slope)\s*(\d+(?:\.\d+)?)\s*%", t):
        if slope_any:
            op, num = slope_any.group(1), float(slope_any.group(2))
        else:
            m = re.search(r"(pendenza|slope)\s*(\d+(?:\.\d+)?)\s*%", t)
            op, num = None, float(m.group(2))
        # interpretazione: capacità minima richiesta
        if (op in (">", ">=")) or re.search(r"\b(almeno|minimo|min)\b", t):
            c["slope_max_percent_min"] = num
        elif op in ("<", "<=") or re.search(r"\b(sotto|max|massimo)\b", t):
            c["slope_max_percent_max"] = num
        else:
            c["slope_max_percent_min"] = num

    return c

# -------------------- Utilities --------------------
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

@st.cache_resource(show_spinner=True)
def load_data_and_index() -> Tuple[pd.DataFrame, List[str], List[Dict[str,Any]], Any, Any]:
    if not DATA_PATH.exists():
        st.error("Manca data/pdp_sample.csv. Caricalo nel repo.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        st.error("Il CSV è vuoto. Aggiungi almeno una riga di prodotto.")
        st.stop()

    passages: List[str] = []
    meta: List[Dict[str, Any]] = []
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
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(passages, normalize_embeddings=True)
    nn = NearestNeighbors(n_neighbors=min(12, len(passages)), metric="cosine")
    nn.fit(emb)
    # persist in session for safety
    st.session_state["__emb_count"] = len(passages)
    return df, passages, meta, model, nn

def retrieve(query: str, model, nn, passages, meta, top_k=10):
    if len(passages) == 0:
        return []
    k = min(top_k, len(passages))
    q_emb = model.encode([query], normalize_embeddings=True)
    dist, idx = nn.kneighbors(q_emb, n_neighbors=k)
    hits = []
    for d, i in zip(dist[0], idx[0]):
        hits.append({"score": float(1 - d), "passage": passages[i], "meta": meta[i]})
    return hits

# -------------------- Hard filter + ranking --------------------
def hard_filter(hits: List[Dict[str,Any]], cons: Dict[str,Any]) -> List[Dict[str,Any]]:
    def ok(m: Dict[str,Any]) -> bool:
        # coverage
        if "coverage_m2_min" in cons and pd.notna(m.get("coverage_m2")):
            if float(m["coverage_m2"]) < cons["coverage_m2_min"]:
                return False
        if "coverage_m2_max" in cons and pd.notna(m.get("coverage_m2")):
            if float(m["coverage_m2"]) > cons["coverage_m2_max"]:
                return False
        # noise
        if "noise_db_max" in cons and pd.notna(m.get("noise_db")):
            if float(m["noise_db"]) > cons["noise_db_max"]:
                return False
        if "noise_db_min" in cons and pd.notna(m.get("noise_db")):
            if float(m["noise_db"]) < cons["noise_db_min"]:
                return False
        # slope
        if "slope_max_percent_min" in cons and pd.notna(m.get("slope_max_percent")):
            if float(m["slope_max_percent"]) < cons["slope_max_percent_min"]:
                return False
        if "slope_max_percent_max" in cons and pd.notna(m.get("slope_max_percent")):
            if float(m["slope_max_percent"]) > cons["slope_max_percent_max"]:
                return False
        return True
    return [h for h in hits if ok(h["meta"])]

def score_fit(h: Dict[str,Any], cons: Dict[str,Any]) -> float:
    fit = 0.0
    m = h["meta"]
    if "coverage_m2_min" in cons and pd.notna(m.get("coverage_m2")):
        w = cons["coverage_m2_min"]; c = float(m["coverage_m2"])
        if c >= w: fit += 1.0 - min((c - w)/max(w,1), 0.5)
    if "noise_db_max" in cons and pd.notna(m.get("noise_db")):
        w = cons["noise_db_max"]; n = float(m["noise_db"])
        if n <= w: fit += 1.0 - min((w - n)/max(w,1), 0.5)
    if "slope_max_percent_min" in cons and pd.notna(m.get("slope_max_percent")):
        w = cons["slope_max_percent_min"]; s = float(m["slope_max_percent"])
        if s >= w: fit += 1.0 - min((s - w)/max(w,1), 0.5)
    return fit

def compose_answer(query: str, hits: List[Dict[str,Any]], cons: Dict[str,Any], filtered_out: bool) -> str:
    if not hits:
        msg = "Nessun modello rispetta i vincoli richiesti."
        tips = []
        if "noise_db_max" in cons: tips.append(f"rumorosità ≤ {cons['noise_db_max']} dB")
        if "coverage_m2_min" in cons: tips.append(f"copertura ≥ {cons['coverage_m2_min']} m²")
        if "slope_max_percent_min" in cons: tips.append(f"pendenza ≥ {cons['slope_max_percent_min']}%")
        if tips: msg += " (vincoli: " + ", ".join(tips) + ")"
        msg += "\nVuoi che ti mostri le **migliori alternative più vicine** ai requisiti?"
        return msg

    for h in hits:
        h["fit"] = score_fit(h, cons)
        h["tot"] = 0.7*h["score"] + 0.3*h["fit"]
    hits = sorted(hits, key=lambda x: x["tot"], reverse=True)

    lines = [f"**Domanda:** {query}"]
    if filtered_out:
        lines.append("> Ho applicato filtri rigidi in base ai vincoli della domanda (es. dB, m², pendenza).")

    lines.append("**Opzioni consigliate (ordinate per pertinenza):**")
    for h in hits:
        m = h["meta"]
        lines.append(
            f"- **{m['title']}** — copertura **{m.get('coverage_m2','?')} m²**, "
            f"pendenza max **{m.get('slope_max_percent','?')}%**, "
            f"rumorosità **{m.get('noise_db','?')} dB** — [PDP]({m['pdp_url']})"
        )
    cites = " | ".join(sorted({h['meta']['pdp_url'] for h in hits if h['meta']['pdp_url']}))
    lines.append(f"\n**Fonti PDP:** {cites}")
    lines.append("\nVuoi **confrontare** due modelli, verificare **compatibilità batteria**, o preferisci quello **più silenzioso**?")
    return "\n".join(lines)

# -------------------- App --------------------
with st.sidebar:
    st.subheader("Suggerimenti")
    st.write("Indica nella domanda vincoli chiari: `600 m²`, `pendenza ≥ 35%`, `sotto 60 dB`.")
    st.divider()
    st.write("Esempi:")
    st.code("robot per 600 m², pendenza 35%, sotto 60 dB")
    st.code("A750: altezza minima di taglio?")
    st.code("compatibilità ePower 20V 4Ah con A1000? (demo)")

df, passages, meta, model, nn = load_data_and_index()

query = st.text_input("Fai una domanda sulle PDP STIGA (IT/EN):", "")
if query:
    cons = parse_constraints(query)
    hits_all = retrieve(query, model, nn, passages, meta, top_k=10)

    filtered = hard_filter(hits_all, cons)
    filtered_out = False
    used = filtered
    if not used:
        used = hits_all  # fallback soft
    else:
        filtered_out = True

    answer = compose_answer(query, used, cons, filtered_out)
    st.markdown(answer)

    with st.expander("🔎 Passaggi usati (debug)"):
        st.write("Vincoli estratti:", cons)
        st.write("Candidati (score, fit, tot):")
        for h in used:
            st.write(h["meta"]["title"], "—", h["meta"]["pdp_url"], f"(score={h.get('score',0):.3f}, fit={h.get('fit',0):.3f}, tot={h.get('tot',0):.3f})")
            st.code(h["passage"])
