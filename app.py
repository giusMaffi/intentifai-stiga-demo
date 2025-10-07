import re
import json
import pandas as pd
import numpy as np
import streamlit as st
import requests

SAFE_BASE = "https://www.stiga.com"
SAFE_LOCALE = "/it"  # puoi cambiare in "/en", "/fr", ecc.
SEARCH_URL = f"{SAFE_BASE}{SAFE_LOCALE}/search/?q="

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
    Estrae vincoli flessibili per ROBOT:
    supporta <, <=, >, >=, 'sotto/max/massimo', 'almeno/minimo/sopra',
    'tra X e Y', e casi come 'copertura di 1000m'.
    Ritorna es: {'coverage_m2': {'max':1000}, 'noise_db': {'max':60}, 'slope_max_percent': {'min':35}}
    """
    t = normalize(q)
    cons: Dict[str, Dict[str, float]] = {}

    # -------- alias & unità --------
    attr_aliases = {
        "coverage_m2": ["m2","mq","m²","m","mt","metri","metro","metriq","copertura","superficie","area","prato"],
        "noise_db": ["db","decibel","rumore","rumorosità","noise"],
        "slope_max_percent": ["%","percento","pendenza","slope"]
    }
    # mappa rapida unit/alias -> attr
    alias_map = {}
    for attr, aliases in attr_aliases.items():
        for a in aliases: alias_map[a] = attr

    def _to_float(s: str) -> float:
        try:
            return float(s.replace(",", "."))
        except:
            return float("nan")

    # helper: prova a capire a quale attributo riferire il numero, guardando unità o il "contesto" vicino
    def unit_to_attr(unit_or_alias: str, window: str) -> str:
        u = (unit_or_alias or "").lower()
        if u in alias_map:
            # special case: 'm' può essere metri lineari; se nel contesto vedo 'copertura/superficie/area/prato' lo tratto come m²
            if u in ["m","mt","metri","metro","metriq"] and not any(w in window for w in ["copertura","superficie","area","prato","mq","m²","m2"]):
                # senza contesto chiaro, NON assumo coverage a tutti i costi -> ritorno None
                return None
            return alias_map[u]
        # nessuna unit: deduco dal contesto (finestra di 24 caratteri intorno al match)
        for kw in ["copertura","superficie","area","prato","mq","m²","m2"]:
            if kw in window: return "coverage_m2"
        for kw in ["db","decibel","rumore","rumorosità","noise"]:
            if kw in window: return "noise_db"
        for kw in ["pendenza","slope","percento","%"]:
            if kw in window: return "slope_max_percent"
        return None

    # -------- 1) operatori espliciti: <=, >=, <, > --------
    # es: "<= 60 db", ">=800 m2", "> 35 %"
    for m in re.finditer(r"(<=|>=|<|>|≤|≥)\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Z%²]*)", t):
        op = m.group(1)
        num = _to_float(m.group(2))
        unit = m.group(3) or ""
        window = t[max(0, m.start()-24): m.end()+24]
        attr = unit_to_attr(unit, window)
        if not attr or np.isnan(num): 
            continue
        cons.setdefault(attr, {})
        if op in ("<","<=","≤"):
            cons[attr]["max"] = num
        elif op in (">",">=","≥"):
            cons[attr]["min"] = num

    # -------- 2) parole-operatore: "sotto/max/massimo ... numero ..." --------
    # permettiamo articoli/preposizioni tra parola e numero: fino a ~10 char non numerici
    for m in re.finditer(r"(sotto|max|massimo|almeno|minimo|sopra|pi[uù]\s*di|maggiore\s*di)\s*[^0-9]{0,10}(\d+(?:[.,]\d+)?)\s*([a-zA-Z%²]*)", t):
        word = m.group(1)
        num = _to_float(m.group(2))
        unit = m.group(3) or ""
        window = t[max(0, m.start()-24): m.end()+24]
        attr = unit_to_attr(unit, window)
        if not attr or np.isnan(num): 
            continue
        cons.setdefault(attr, {})
        if word in ["sotto","max","massimo"]:
            cons[attr]["max"] = num
        elif word in ["almeno","minimo","sopra","più di","piu di","maggiore di"]:
            cons[attr]["min"] = num

    # -------- 3) range: "tra 55 e 60 db" / "fra 500 e 800 m2" --------
    for m in re.finditer(r"(tra|fra)\s*(\d+(?:[.,]\d+)?)\s*(?:e|ed)\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Z%²]*)", t):
        a = _to_float(m.group(2)); b = _to_float(m.group(3))
        lo, hi = min(a,b), max(a,b)
        unit = m.group(4) or ""
        window = t[max(0, m.start()-24): m.end()+24]
        attr = unit_to_attr(unit, window)
        if not attr or np.isnan(lo) or np.isnan(hi): 
            continue
        cons.setdefault(attr, {})
        cons[attr]["min"] = lo
        cons[attr]["max"] = hi

    # -------- 4) pattern specifici di copertura: "copertura di 1000m / 1000 mq / 1000" --------
    # Se troviamo 'copertura/superficie/area/prato ... numero [m/mq/m²]' assumiamo coverage_m2
    for m in re.finditer(r"(copertura|superficie|area|prato)[^0-9]{0,10}(\d+(?:[.,]\d+)?)(\s*(m2|mq|m²|m|mt|metri)?)", t):
        num = _to_float(m.group(2))
        unit = (m.group(4) or "").lower()
        if np.isnan(num): 
            continue
        cons.setdefault("coverage_m2", {})
        # Se c'è "sotto/max/massimo" nelle vicinanze, trattalo come max; altrimenti come min (richiesta minima)
        window = t[max(0, m.start()-16): m.end()+16]
        if any(w in window for w in ["sotto","max","massimo"]):
            cons["coverage_m2"]["max"] = num
        else:
            cons["coverage_m2"]["min"] = num

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
@st.cache_data(show_spinner=False, ttl=3600)
def url_is_ok(url: str) -> bool:
    if not url or not url.startswith("http"):
        return False
    try:
        # alcuni siti non rispondono a HEAD, usiamo GET leggero
        r = requests.get(url, timeout=3, allow_redirects=True, stream=True)
        return 200 <= r.status_code < 400
    except Exception:
        return False

def build_search_fallback(m: Dict[str, any]) -> str:
    q = m.get("sku") or m.get("title") or ""
    q = requests.utils.quote(str(q))
    return f"{SEARCH_URL}{q}"

def safe_pdp_url(m: Dict[str, any]) -> str:
    url = m.get("pdp_url") or ""
    if url_is_ok(url):
        return url
    return ""  # segnala che non è disponibile

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
        cites.add(m.get("pdp_url",""))
        bits = []
        if "coverage_m2" in key_fields: bits.append(f"copertura **{m.get('coverage_m2','?')} m²**")
        if "slope_max_percent" in key_fields: bits.append(f"pendenza max **{m.get('slope_max_percent','?')}%**")
        if "noise_db" in key_fields: bits.append(f"rumorosità **{m.get('noise_db','?')} dB**")

        pdp = safe_pdp_url(m)
        if pdp:
            link_part = f"[PDP]({pdp}) · [Cerca su STIGA]({build_search_fallback(m)})"
        else:
            link_part = f"[Cerca su STIGA]({build_search_fallback(m)})"

        st.markdown(f"- **{m['title']}** — " + " · ".join(bits) + " — " + link_part)

        shown += 1
        if shown >= limit:
            break

    cites = {c for c in cites if c}
    if cites:
        st.markdown("**Fonti PDP (fornite):** " + " | ".join(sorted(cites)))

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
