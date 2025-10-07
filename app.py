import re
import json
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

# ============ Config ============
DATA_PATH = Path("data/pdp_sample.csv")

CATEGORY_CONFIG = {
    "robot": {
        "aliases": ["robot", "robot tagliaerba", "robot lawnmower", "tosaerba robot", "robot per prato", "lawn"],
        "attributes": ["coverage_m2", "noise_db", "slope_max_percent"],
        "fit_weights": {"coverage_m2": 0.4, "noise_db": 0.3, "slope_max_percent": 0.3},
        "label": "Robot Lawnmowers",
        "result_fields": ["coverage_m2", "slope_max_percent", "noise_db"],
    },
    # pronto per future categorie (chainsaw, tosaerba, ecc.)
}

st.set_page_config(page_title="IntentifAI × STIGA – PDP Chat (Robot)", page_icon="🟡", layout="wide")
st.title("Assistente prodotti – Robot STIGA (Demo)")
st.caption("Demo interna basata su dati PDP pubblici. Risposte con fonti. UI senza reasoning esposto.")

# ============ Helpers ============

def normalize(text: str) -> str:
    return (text or "").lower().strip()

def auto_detect_category(q: str) -> str:
    t = normalize(q)
    for cat, cfg in CATEGORY_CONFIG.items():
        if any(a in t for a in cfg["aliases"]):
            return cat
    return "robot"  # default

def parse_constraints(q: str) -> Dict[str, Any]:
    """ Estrae vincoli generici (coverage m², noise dB, slope %) interpretati per ROBOT. """
    t = normalize(q).replace(",", ".")
    c: Dict[str, Any] = {}

    # coverage (m²)
    cov = re.search(r"(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)\s*(m2|mq|m²)", t)
    if cov:
        op, num = cov.group(1), float(cov.group(2))
        if op in (">", ">=") or re.search(r"\b(almeno|minimo|min)\b", t):
            c["coverage_m2_min"] = num
        elif op in ("<", "<=") or re.search(r"\b(sotto|max|massimo)\b", t):
            c["coverage_m2_max"] = num
        else:
            c["coverage_m2_min"] = num

    # noise (dB)
    noisep = re.search(r"(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)\s*d\s*b", t)
    if noisep or re.search(r"(sotto|max|massimo|minimo|almeno)\s*\d+(?:\.\d+)?\s*d\s*b", t):
        if noisep:
            op, num = noisep.group(1), float(noisep.group(2))
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

    # slope (%)
    slope = re.search(r"(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)\s*%(\s*(pendenza|slope))?", t)
    if slope or re.search(r"(pendenza|slope)\s*(\d+(?:\.\d+)?)\s*%", t):
        if slope:
            op, num = slope.group(1), float(slope.group(2))
        else:
            m = re.search(r"(pendenza|slope)\s*(\d+(?:\.\d+)?)\s*%", t)
            op, num = None, float(m.group(2))
        # per robot: pendenza richiesta è capacità minima
        if (op in (">", ">=")) or re.search(r"\b(almeno|minimo|min)\b", t):
            c["slope_max_percent_min"] = num
        elif op in ("<", "<=") or re.search(r"\b(sotto|max|massimo)\b", t):
            c["slope_max_percent_max"] = num
        else:
            c["slope_max_percent_min"] = num

    return c

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
def load_data_and_embeddings() -> Tuple[pd.DataFrame, List[str], List[Dict[str,Any]], Any, np.ndarray]:
    if not DATA_PATH.exists():
        st.error("Manca data/pdp_sample.csv. Carica il file nel repo.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        st.error("Il CSV è vuoto. Aggiungi almeno un prodotto.")
        st.stop()

    passages: List[str] = []
    meta: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        passages.append(build_passage(r))
        meta.append({
            "sku": r.get("sku",""),
            "title": r.get("title",""),
            "pdp_url": r.get("pdp_url",""),
            "category": normalize(r.get("category","")),
            "coverage_m2": r.get("coverage_m2", np.nan),
            "cut_min": r.get("cutting_height_min_mm", np.nan),
            "cut_max": r.get("cutting_height_max_mm", np.nan),
            "noise_db": r.get("noise_db", np.nan),
            "slope_max_percent": r.get("slope_max_percent", np.nan),
            "price_eur": r.get("price_eur", np.nan),
        })

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb_matrix = model.encode(passages, normalize_embeddings=True)  # (N,D), normalizzati
    return df, passages, meta, model, emb_matrix

def cosine_scores_to_all(q: str, model, emb_matrix: np.ndarray) -> np.ndarray:
    q_emb = model.encode([q], normalize_embeddings=True)  # (1,D)
    return (emb_matrix @ q_emb.T).ravel()  # cos perché embeddings normalizzati

# ---------- Filtri + ranking ----------
def category_mask(meta: List[Dict[str,Any]], cat_key: str) -> List[int]:
    """Ritorna indici dei prodotti che appartengono alla categoria selezionata."""
    return [i for i, m in enumerate(meta) if normalize(m.get("category","")) == normalize(cat_key)]

def hard_filter_indices_robot(meta: List[Dict[str,Any]], cons: Dict[str,Any], idx_pool: List[int]) -> List[int]:
    out = []
    for i in idx_pool:
        m = meta[i]
        ok = True
        # coverage
        if "coverage_m2_min" in cons and pd.notna(m.get("coverage_m2")):
            ok &= float(m["coverage_m2"]) >= cons["coverage_m2_min"]
        if "coverage_m2_max" in cons and pd.notna(m.get("coverage_m2")):
            ok &= float(m["coverage_m2"]) <= cons["coverage_m2_max"]
        # noise
        if "noise_db_max" in cons and pd.notna(m.get("noise_db")):
            ok &= float(m["noise_db"]) <= cons["noise_db_max"]
        if "noise_db_min" in cons and pd.notna(m.get("noise_db")):
            ok &= float(m["noise_db"]) >= cons["noise_db_min"]
        # slope
        if "slope_max_percent_min" in cons and pd.notna(m.get("slope_max_percent")):
            ok &= float(m["slope_max_percent"]) >= cons["slope_max_percent_min"]
        if "slope_max_percent_max" in cons and pd.notna(m.get("slope_max_percent")):
            ok &= float(m["slope_max_percent"]) <= cons["slope_max_percent_max"]
        if ok:
            out.append(i)
    return out

def score_fit_robot(m: Dict[str,Any], cons: Dict[str,Any], weights: Dict[str,float]) -> float:
    fit = 0.0
    # coverage (preferisci >= richiesto e vicino al target)
    if "coverage_m2_min" in cons and pd.notna(m.get("coverage_m2")):
        w = cons["coverage_m2_min"]; c = float(m["coverage_m2"])
        if c >= w:
            fit += weights.get("coverage_m2", 0.0) * (1.0 - min((c - w)/max(w,1), 0.5))
    if "coverage_m2_max" in cons and pd.notna(m.get("coverage_m2")):
        w = cons["coverage_m2_max"]; c = float(m["coverage_m2"])
        if c <= w:
            fit += weights.get("coverage_m2", 0.0) * (1.0 - min((w - c)/max(w,1), 0.5))
    # noise (preferisci <= soglia)
    if "noise_db_max" in cons and pd.notna(m.get("noise_db")):
        w = cons["noise_db_max"]; n = float(m["noise_db"])
        if n <= w:
            fit += weights.get("noise_db", 0.0) * (1.0 - min((w - n)/max(w,1), 0.5))
    if "noise_db_min" in cons and pd.notna(m.get("noise_db")):
        w = cons["noise_db_min"]; n = float(m["noise_db"])
        if n >= w:
            fit += weights.get("noise_db", 0.0) * (1.0 - min((n - w)/max(w,1), 0.5))
    # slope (preferisci >= richiesta)
    if "slope_max_percent_min" in cons and pd.notna(m.get("slope_max_percent")):
        wv = cons["slope_max_percent_min"]; s = float(m["slope_max_percent"])
        if s >= wv:
            fit += weights.get("slope_max_percent", 0.0) * (1.0 - min((s - wv)/max(wv,1), 0.5))
    if "slope_max_percent_max" in cons and pd.notna(m.get("slope_max_percent")):
        wv = cons["slope_max_percent_max"]; s = float(m["slope_max_percent"])
        if s <= wv:
            fit += weights.get("slope_max_percent", 0.0) * (1.0 - min((wv - s)/max(wv,1), 0.5))
    return fit

def render_results(title: str, order_idx: List[int], meta: List[Dict[str,Any]], cfg: Dict[str,Any], limit: int = 50):
    if not order_idx:
        st.write("Nessun modello corrisponde ai criteri indicati. Puoi modificare m², dB o pendenza.")
        return
    st.subheader(title)
    cites = set()
    shown = 0
    key_fields = cfg.get("result_fields", [])
    for i in order_idx:
        m = meta[i]
        cites.add(m["pdp_url"])
        st.markdown(
            f"- **{m['title']}** — "
            + " · ".join([
                f"copertura **{m.get('coverage_m2','?')} m²**" if "coverage_m2" in key_fields else "",
                f"pendenza max **{m.get('slope_max_percent','?')}%**" if "slope_max_percent" in key_fields else "",
                f"rumorosità **{m.get('noise_db','?')} dB**" if "noise_db" in key_fields else "",
            ]).replace("  ·  · ", " · ").strip(" · ")
            + f" — [PDP]({m['pdp_url']})"
        )
        shown += 1
        if shown >= limit:
            break
    st.markdown("**Fonti PDP:** " + " | ".join(sorted(cites)))

# ============ App ============

with st.sidebar:
    st.subheader("Filtri (mettili nella domanda)")
    st.write("Esempi: `600 m²`, `pendenza ≥ 35%`, `sotto 60 dB`")
    cat_choice = st.selectbox("Categoria", options=["Auto (dalla domanda)", "Robot"], index=1)

df, passages, meta, model, emb_matrix = load_data_and_embeddings()

query = st.text_input("Fai una domanda sui robot STIGA (IT/EN):", "")

if query:
    # Categoria
    target_cat = "robot" if cat_choice == "Robot" else auto_detect_category(query)
    cfg = CATEGORY_CONFIG[target_cat]

    # similarità verso tutto il dataset
    cos = cosine_scores_to_all(query, model, emb_matrix)  # (N,)

    # pool della categoria
    pool_idx = category_mask(meta, target_cat)

    # vincoli dalla query
    cons = parse_constraints(query)

    # hard filter AND su TUTTI i vincoli per la categoria Robot
    idx_valid = hard_filter_indices_robot(meta, cons, pool_idx)

    # se ci sono match esatti → ordino
    if idx_valid:
        scored = []
        for i in idx_valid:
            fit = score_fit_robot(meta[i], cons, cfg["fit_weights"])
            tot = 0.75*float(cos[i]) + 0.25*fit
            scored.append((tot, i))
        order_idx = [i for _, i in sorted(scored, key=lambda x: x[0], reverse=True)]
        render_results("Opzioni adatte alle tue esigenze", order_idx, meta, cfg)
    else:
        # fallback: nessun match preciso → mostro i più pertinenti della categoria
        alt = [(float(cos[i]), i) for i in pool_idx]
        order_idx = [i for _, i in sorted(alt, key=lambda x: x[0], reverse=True)]
        render_results("Nessun match perfetto — alternative più vicine", order_idx, meta, cfg)

    # Debug SOLO per staff (non per clienti)
    with st.expander("🔧 Debug (staff)"):
        st.write("Categoria:", target_cat)
        st.write("Vincoli estratti:", cons)
        st.write("Prodotti categoria:", len(pool_idx))
        st.write("Match precisi:", len(idx_valid))
