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

# Categoria: ROBOT (scalabile a future categorie)
CATEGORY_CONFIG = {
    "robot": {
        "label": "Robot Lawnmowers",
        # attributi chiave con metadati per parsing/filtri
        "attributes": {
            "coverage_m2": {
                "aliases": ["m2", "mq", "m²", "copertura", "superficie", "area", "metri quadrati"],
                "unit": "m2",
                "prefer_direction": "min"  # spesso l'utente indica un minimo richiesto, ma supportiamo tutto
            },
            "noise_db": {
                "aliases": ["db", "decibel", "rumore", "rumorosità", "noise"],
                "unit": "db",
                "prefer_direction": "max"
            },
            "slope_max_percent": {
                "aliases": ["pendenza", "slope", "%"],
                "unit": "%",
                "prefer_direction": "min"
            }
        },
        # pesi per ranking "fit" (multi-criterio)
        "fit_weights": {"coverage_m2": 0.4, "noise_db": 0.3, "slope_max_percent": 0.3},
        # campi mostrati nella riga risultato
        "result_fields": ["coverage_m2", "slope_max_percent", "noise_db"],
    }
}

st.set_page_config(page_title="Assistente prodotti – Robot STIGA (Demo)", page_icon="🟡", layout="wide")
st.title("Assistente prodotti – Robot STIGA (Demo)")
st.caption("Demo interna basata su dati PDP pubblici. UI pulita, risultati con fonti.")

# ============ Helpers ============

def normalize(text: str) -> str:
    return (text or "").lower().strip()

def auto_detect_category(q: str) -> str:
    # per ora abbiamo solo robot
    return "robot"

# -------- parsing generico di range e operatori --------
OPS = {
    "<": "max", "<=": "max", "≤": "max",
    ">": "min", ">=": "min", "≥": "min"
}
WORDS_MAX = ["sotto", "max", "massimo", "non oltre", "fino a"]
WORDS_MIN = ["sopra", "almeno", "minimo", "da", "più di", "maggiore di"]
WORDS_ABOUT = ["circa", "intorno a", "~", "approx", "approssimativamente"]
WORDS_BETWEEN = ["tra", "fra"]  # es. "tra 55 e 60 dB"

def _to_float(s: str) -> float:
    try:
        return float(s.replace(",", "."))
    except:
        return float("nan")

def parse_constraints_flexible(q: str, cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Ritorna un dict del tipo:
    {
      "coverage_m2": {"min": 600},            # o {"max": 700}, o {"min": 600, "max": 900}
      "noise_db": {"max": 60},
      "slope_max_percent": {"min": 35}
    }
    Capisce: <, <=, >, >=, "sotto", "sopra", "almeno", "massimo", "tra X e Y", "circa N".
    """
    t = normalize(q)
    cons: Dict[str, Dict[str, float]] = {}

    # pre-build alias→attr map
    alias_map = {}
    for attr, meta in cfg["attributes"].items():
        for a in meta["aliases"]:
            alias_map[a] = attr

    # 1) pattern con operatore esplicito: es. "<= 60 dB", ">= 600 m2"
    #    catturiamo (op)(numero)(unit o alias)
    #    unit può essere db, %, m2/mq/m²; oppure usiamo alias parola (pendenza, rumorosità...)
    pattern_op = r"(<=|>=|<|>|≤|≥)\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Z%²]+)?"
    for m in re.finditer(pattern_op, t):
        op_raw = m.group(1)
        num = _to_float(m.group(2))
        unit_or_alias = (m.group(3) or "").lower()
        # mappa unit/alias -> attr
        attr = None
        if unit_or_alias in ["m2", "mq", "m²"]:
            attr = "coverage_m2"
        elif unit_or_alias in ["db", "decibel"]:
            attr = "noise_db"
        elif unit_or_alias in ["%"]:
            attr = "slope_max_percent"
        elif unit_or_alias in alias_map:
            attr = alias_map[unit_or_alias]
        # se attr non individuato qui, proviamo a dedurre dall'intorno testi (grezzo ma utile)
        if not attr:
            window = t[max(0, m.start()-20):m.end()+20]
            for al, a_attr in alias_map.items():
                if al in window:
                    attr = a_attr
                    break
        if not attr:
            continue
        # scrivi min/max
        cons.setdefault(attr, {})
        direction = OPS.get(op_raw, None)
        if direction == "max":
            cons[attr]["max"] = num
        elif direction == "min":
            cons[attr]["min"] = num

    # 2) frasi "sotto/sopra/almeno/massimo <numero> <unit/alias>"
    pattern_word = r"(sotto|max|massimo|almeno|minimo|sopra|più di|maggiore di)\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Z%²]+)?"
    for m in re.finditer(pattern_word, t):
        word = m.group(1)
        num = _to_float(m.group(2))
        unit_or_alias = (m.group(3) or "").lower()
        attr = None
        if unit_or_alias in ["m2", "mq", "m²"]:
            attr = "coverage_m2"
        elif unit_or_alias in ["db", "decibel"]:
            attr = "noise_db"
        elif unit_or_alias in ["%"]:
            attr = "slope_max_percent"
        elif unit_or_alias in alias_map:
            attr = alias_map[unit_or_alias]
        if not attr:
            # prova a dedurre dall'intorno
            span = m.span()
            window = t[max(0, span[0]-20):span[1]+20]
            for al, a_attr in alias_map.items():
                if al in window:
                    attr = a_attr
                    break
        if not attr:
            continue
        cons.setdefault(attr, {})
        if word in WORDS_MAX:
            cons[attr]["max"] = num
        elif word in WORDS_MIN or word in ["più di", "maggiore di"]:
            cons[attr]["min"] = num

    # 3) range: "tra X e Y <unit/alias>"
    #    esempi: "tra 55 e 60 dB", "fra 500 e 800 m2"
    pattern_between = r"(tra|fra)\s*(\d+(?:[.,]\d+)?)\s*e\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Z%²]+)?"
    for m in re.finditer(pattern_between, t):
        a = _to_float(m.group(2))
        b = _to_float(m.group(3))
        unit_or_alias = (m.group(4) or "").lower()
        lo, hi = min(a, b), max(a, b)
        attr = None
        if unit_or_alias in ["m2", "mq", "m²"]:
            attr = "coverage_m2"
        elif unit_or_alias in ["db", "decibel"]:
            attr = "noise_db"
        elif unit_or_alias in ["%"]:
            attr = "slope_max_percent"
        elif unit_or_alias in alias_map:
            attr = alias_map[unit_or_alias]
        if attr:
            cons.setdefault(attr, {})
            cons[attr]["min"] = lo
            cons[attr]["max"] = hi

    # 4) "circa N <unit>" -> min/max con piccola tolleranza (±5% o soglia fissa)
    pattern_about = r"(circa|intorno a|~|approx|approssimativamente)\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Z%²]+)?"
    for m in re.finditer(pattern_about, t):
        num = _to_float(m.group(2))
        unit_or_alias = (m.group(3) or "").lower()
        attr = None
        if unit_or_alias in ["m2", "mq", "m²"]:
            attr = "coverage_m2"
        elif unit_or_alias in ["db", "decibel"]:
            attr = "noise_db"
        elif unit_or_alias in ["%"]:
            attr = "slope_max_percent"
        elif unit_or_alias in alias_map:
            attr = alias_map[unit_or_alias]
        if not attr:
            continue
        cons.setdefault(attr, {})
        tol = 0.05 * num  # ±5%
        cons[attr]["min"] = min(cons[attr].get("min", num - tol), num - tol)
        cons[attr]["max"] = max(cons[attr].get("max", num + tol), num + tol)

    return cons

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
    emb_matrix = model.encode(passages, normalize_embeddings=True)  # (N,D)
    return df, passages, meta, model, emb_matrix

def cosine_scores_to_all(q: str, model, emb_matrix: np.ndarray) -> np.ndarray:
    q_emb = model.encode([q], normalize_embeddings=True)  # (1,D)
    return (emb_matrix @ q_emb.T).ravel()  # cos perché normalizzati

# ---------- Filtri + ranking ----------
def category_mask(meta: List[Dict[str,Any]], cat_key: str) -> List[int]:
    return [i for i, m in enumerate(meta) if normalize(m.get("category","")) == normalize(cat_key)]

def within_range(val: float, rng: Dict[str,float]) -> bool:
    if pd.isna(val): 
        return False
    ok = True
    if "min" in rng:
        ok &= float(val) >= rng["min"]
    if "max" in rng:
        ok &= float(val) <= rng["max"]
    return ok

def hard_filter_indices_cat(meta: List[Dict[str,Any]], cat_cfg: Dict[str,Any], cons: Dict[str,Dict[str,float]], idx_pool: List[int]) -> List[int]:
    out = []
    for i in idx_pool:
        m = meta[i]
        ok = True
        for attr in cat_cfg["attributes"].keys():
            if attr in cons:
                rng = cons[attr]
                ok &= within_range(m.get(attr if attr != "slope_max_percent" else "slope_max_percent"), rng)
        if ok:
            out.append(i)
    return out

def score_fit_cat(m: Dict[str,Any], cons: Dict[str,Dict[str,float]], weights: Dict[str,float]) -> float:
    fit = 0.0
    for attr, rng in cons.items():
        val = m.get(attr)
        if pd.isna(val): 
            continue
        w = weights.get(attr, 0.0)
        # distanza dal bordo "migliore": se "max", preferisci più basso; se "min", preferisci più alto;
        # se range, preferisci stare vicino al centro del range
        if "min" in rng and "max" in rng:
            center = (rng["min"] + rng["max"]) / 2.0
            span = max(rng["max"] - rng["min"], 1e-6)
            diff = abs(float(val) - center) / span  # 0 = centro perfetto
            fit += w * (1.0 - min(diff, 1.0))  # più vicino al centro, meglio
        elif "max" in rng and "min" not in rng:
            # più basso è, meglio è; normalizza rispetto alla soglia
            if float(val) <= rng["max"]:
                fit += w * (1.0 - min((rng["max"] - float(val)) / max(rng["max"], 1), 0.5))
        elif "min" in rng and "max" not in rng:
            # più alto è, meglio è; normalizza rispetto alla soglia
            if float(val) >= rng["min"]:
                fit += w * (1.0 - min((float(val) - rng["min"]) / max(rng["min"], 1), 0.5))
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
        bits = []
        if "coverage_m2" in key_fields: bits.append(f"copertura **{m.get('coverage_m2','?')} m²**")
        if "slope_max_percent" in key_fields: bits.append(f"pendenza max **{m.get('slope_max_percent','?')}%**")
        if "noise_db" in key_fields: bits.append(f"rumorosità **{m.get('noise_db','?')} dB**")
        st.markdown(f"- **{m['title']}** — " + " · ".join(bits) + f" — [PDP]({m['pdp_url']})")
        shown += 1
        if shown >= limit:
            break
    st.markdown("**Fonti PDP:** " + " | ".join(sorted(cites)))

# ============ App ============

with st.sidebar:
    st.subheader("Suggerimenti")
    st.write("Esempi: `≥ 800 m²`, `sotto 60 dB`, `pendenza ≥ 35%`, `tra 55 e 60 dB`")
    cat_choice = st.selectbox("Categoria", options=["Robot"], index=0)

df, passages, meta, model, emb_matrix = load_data_and_embeddings()

query = st.text_input("Fai una domanda sui robot STIGA (IT/EN):", "")

if query:
    target_cat = "robot"
    cfg = CATEGORY_CONFIG[target_cat]

    # similarità verso tutto il dataset
    cos = cosine_scores_to_all(query, model, emb_matrix)  # (N,)

    # pool della categoria
    pool_idx = category_mask(meta, target_cat)

    # vincoli (flessibili): <, >, tra, circa...
    cons = parse_constraints_flexible(query, cfg)

    # hard filter combinato (AND) su TUTTI i vincoli passati
    idx_valid = hard_filter_indices_cat(meta, cfg, cons, pool_idx)

    if idx_valid:
        scored = []
        for i in idx_valid:
            fit = score_fit_cat(meta[i], cons, cfg["fit_weights"])
            tot = 0.75*float(cos[i]) + 0.25*fit
            scored.append((tot, i))
        order_idx = [i for _, i in sorted(scored, key=lambda x: x[0], reverse=True)]
        render_results("Opzioni adatte alle tue esigenze", order_idx, meta, cfg)
    else:
        # fallback: niente match perfetto → alternative per pertinenza nella categoria
        alt = [(float(cos[i]), i) for i in pool_idx]
        order_idx = [i for _, i in sorted(alt, key=lambda x: x[0], reverse=True)]
        render_results("Nessun match perfetto — alternative più vicine", order_idx, meta, cfg)

    # Debug interno per noi
    with st.expander("🔧 Debug (staff)"):
        st.write("Vincoli estratti:", cons)
        st.write("Prodotti categoria:", len(pool_idx))
        st.write("Match precisi:", len(idx_valid))
