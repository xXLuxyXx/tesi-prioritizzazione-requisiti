import json
import os
import re
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OrdinalEncoder

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except Exception:
    _NLP = None

# Caricamento artefatti legacy (mantenuti per retrocompatibilità solo se presenti)
pca_emb = joblib.load("models/pca.pkl") if os.path.exists("models/pca.pkl") else None
scaler = joblib.load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None

# Set identico a run_ablation_study_2.py (Sez. 3.1.1)
DEONTIC_AND_NEG = {
    "must", "shall", "should", "may", "might", "can", "could", "will", "would",
    "not", "no", "never", "none", "cannot", "without", "before", "after", "within"
}
CUSTOM_STOPWORDS = (_NLP.Defaults.stop_words - DEONTIC_AND_NEG) if _NLP is not None else set()

ACRONYM_MAP = {
    r"\bauth\b": "authentication",
    r"\bdb\b": "database",
    r"\bperf\b": "performance",
    r"\breq\b": "requirement",
    r"\bconfig\b": "configuration",
    r"\bui\b": "user interface",
    r"\bapi\b": "application programming interface",
    r"\bsec\b": "security"
}

UI_TO_DATASET = {
    "functional": "fr",
    "non-functional": "nfr",
    "non functional": "nfr",
    "quality": "qr"
}

CAT_COLS = ["Type", "Change", "Urgency", "Origin"]
_REF_EMBEDDINGS_CACHE = None


def _normalize_cat_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().replace(UI_TO_DATASET)


def _build_fitted_cat_encoder() -> OrdinalEncoder:
    """
    Ricostruisce esattamente cat_enc (OrdinalEncoder) usato in run_ablation_study_2.py.
    """
    frames = []
    for path in ("dataset/full_requirements.csv", "dataset/groq_augmented_train.csv"):
        if os.path.exists(path):
            df_tmp = pd.read_csv(path, usecols=lambda c: c in CAT_COLS)
            if set(CAT_COLS).issubset(df_tmp.columns):
                frames.append(df_tmp[CAT_COLS])

    if frames:
        df_cat = pd.concat(frames, ignore_index=True)
    else:
        df_cat = pd.DataFrame({
            "Type": ["fr", "nfr", "qr", "fr", "nfr"],
            "Change": ["addition", "deletion", "modification", "addition", "addition"],
            "Urgency": ["later", "soon", "later", "soon", "later"],
            "Origin": ["customer", "developer knowledge", "market", "organization", "project vision"],
        })

    for col in CAT_COLS:
        df_cat[col] = _normalize_cat_series(df_cat[col])

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(df_cat[CAT_COLS])
    return enc


_CAT_ENCODER = _build_fitted_cat_encoder()


def clean_requirement_text(text: str) -> str:
    """Sanificazione sintattica domain-agnostic identica a run_ablation_study_2.py."""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[?[A-Z]{2,6}[-_]\d+\]?:?", " ", text)
    text = re.sub(r"#\d+", " ", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(ms|sec|seconds|minutes|min|hours|s)\b", " <TIME_CONSTRAINT> ", text, flags=re.I)
    text = re.sub(r"\b\d+(\.\d+)?\s*%", " <PERCENTAGE_THRESHOLD> ", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(mb|gb|kb|requests/s|req/s|tps)\b", " <CAPACITY_THRESHOLD> ", text, flags=re.I)
    text = text.lower()

    for pat, repl in ACRONYM_MAP.items():
        text = re.sub(pat, repl, text)

    text = re.sub(r"[^a-z0-9_<>\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if _NLP is not None and text:
        doc = _NLP(text)
        tokens = []
        for tok in doc:
            if tok.text.startswith("<") and tok.text.endswith(">"):
                tokens.append(tok.text)
            elif tok.text in DEONTIC_AND_NEG:
                tokens.append(tok.text)
            elif tok.text not in CUSTOM_STOPWORDS and len(tok.text) > 1:
                tokens.append(tok.lemma_)
        return " ".join(tokens)

    return text


def load_optimized_artifacts():
    """Carica il classificatore calibrato ottimizzato, il target encoder e opt_config.json."""
    clf_opt = joblib.load("models/classifier_optimized.pkl")
    target_enc = joblib.load("models/target_encoder_optimized.pkl")
    opt_cfg = {"theta_high": 0.44, "k_pca": 24, "entropy_threshold": 1.15}
    if os.path.exists("models/opt_config.json"):
        with open("models/opt_config.json", "r", encoding="utf-8") as f:
            opt_cfg.update(json.load(f))
    return clf_opt, target_enc, opt_cfg


def compute_shannon_entropy(proba: np.ndarray) -> float:
    """Calcola l'entropia di Shannon H(Y | z_i) in bit."""
    p = np.clip(proba, 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def preprocess_input(row_dict):
    """Pipeline legacy (retrocompatibilità)."""
    from scripts.preprocessing import scale, encode, embed
    df = pd.DataFrame([row_dict])
    scaled_num = scale(df, scaler=scaler)
    encoded_cat = encode(df)
    emb = embed(df)
    emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    emb_reduced = pca_emb.transform(emb_norm)
    X = np.hstack([emb_reduced * 0.5, encoded_cat.astype(float), scaled_num])
    return X[0]


def _get_default_reference_embeddings(sbert_model: SentenceTransformer) -> np.ndarray | None:
    """Calcola e mette in cache gli embedding del dataset di training per l'inferenza singola."""
    global _REF_EMBEDDINGS_CACHE
    if _REF_EMBEDDINGS_CACHE is not None:
        return _REF_EMBEDDINGS_CACHE

    texts = []
    for path in ("dataset/full_requirements.csv", "dataset/groq_augmented_train.csv"):
        if os.path.exists(path):
            df_tmp = pd.read_csv(path, usecols=lambda c: c == "Requirement")
            if "Requirement" in df_tmp.columns:
                texts.extend(df_tmp["Requirement"].dropna().astype(str).tolist())

    if not texts:
        return None

    cleaned = [clean_requirement_text(t) for t in texts]
    embs = sbert_model.encode(cleaned, show_progress_bar=False)
    norms = np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
    _REF_EMBEDDINGS_CACHE = embs / norms
    return _REF_EMBEDDINGS_CACHE


def _compute_similarity_stats(
    emb_vec: np.ndarray,
    reference_embeddings: np.ndarray | None = None,
    graph_metrics: dict | None = None
) -> tuple[float, float, float]:
    """
    Calcola deg_centrality, max_sim, mean_top3 come in extract_struct_features().
    """
    gm = graph_metrics or {}
    if reference_embeddings is not None and len(reference_embeddings) > 0:
        sims = cosine_similarity(emb_vec.reshape(1, -1), reference_embeddings)[0]
        sims = sims[sims < 0.9999]
        if len(sims) > 0:
            deg_centrality = float((sims > 0.55).sum() / len(sims))
            max_sim = float(np.max(sims))
            top_k = np.sort(sims)[-min(3, len(sims)):]
            mean_top3 = float(np.mean(top_k))
            return deg_centrality, max_sim, mean_top3

    deg_centrality = float(gm.get("deg_centrality", 0.0))
    max_sim = float(gm.get("max_sim", 0.0))
    mean_top3 = float(gm.get("mean_top3", 0.0))
    return deg_centrality, max_sim, mean_top3


def predict_priority_optimized(
    row_dict: dict,
    sbert_model: SentenceTransformer,
    clf_opt,
    target_enc,
    opt_cfg: dict,
    graph_metrics: dict | None = None,
    reference_embeddings: np.ndarray | None = None,
) -> dict:
    """
    Esegue l'inferenza completa con la pipeline ottimizzata (Config. D):
    pulizia domain-agnostic -> Dense Embedding L2 (384D) ->
    4 categorical OrdinalEncoded + 8 structural/graph features (384..395) ->
    probabilità calibrate -> soglia theta_high -> entropia di Shannon.
    """
    raw_text = str(row_dict.get("Requirement", ""))
    cleaned_text = clean_requirement_text(raw_text)

    # 1. Dense Sentence Embedding normalizzato L2 (indici 0..383)
    emb = sbert_model.encode([cleaned_text], show_progress_bar=False)[0]
    emb = emb / max(float(np.linalg.norm(emb)), 1e-12)

    if reference_embeddings is None:
        reference_embeddings = _get_default_reference_embeddings(sbert_model)

    # 2. Colonne categoriche codificate con lo stesso OrdinalEncoder (indici 384..387)
    cat_df = pd.DataFrame([{
        "Type": row_dict.get("Type", "fr"),
        "Change": row_dict.get("Change", "addition"),
        "Urgency": row_dict.get("Urgency", "soon"),
        "Origin": row_dict.get("Origin", "organization"),
    }])
    for col in CAT_COLS:
        cat_df[col] = _normalize_cat_series(cat_df[col])
    cat_encoded = _CAT_ENCODER.transform(cat_df[CAT_COLS])[0]  # shape (4,)

    # 3. Feature numeriche, topologiche e lessicali (indici 388..395)
    n_deps = float(row_dict.get("N_dependencies", 0) or 0)
    cost = float(row_dict.get("Cost", 0) or 0)
    time_val = float(row_dict.get("Time", 0) or 0)

    deg_centrality, max_sim, mean_top3 = _compute_similarity_stats(
        emb, reference_embeddings=reference_embeddings, graph_metrics=graph_metrics
    )
    req_len = float(len(raw_text.split()))
    has_deontic = 1.0 if re.search(r"\b(must|shall|never|cannot)\b", raw_text.lower()) else 0.0

    struct_vec = np.array([
        n_deps,          # 388
        cost,            # 389
        time_val,        # 390
        deg_centrality,  # 391
        max_sim,         # 392
        mean_top3,       # 393
        req_len,         # 394
        has_deontic      # 395
    ], dtype=float)

    # Vettore finale (1, 396) allineato a ColumnTransformer(["sem": 0..383, "struct": 384..395])
    X_input = np.hstack([emb, cat_encoded, struct_vec]).reshape(1, -1)

    # 4. Probabilità posteriori calibrate
    proba = clf_opt.predict_proba(X_input)[0]
    if hasattr(target_enc, "classes_"):
        raw_labels = target_enc.classes_
    else:
        raw_labels = target_enc.categories_[0]
    classes_raw = [str(c).lower() for c in raw_labels]
    proba_map = {cls_name: float(p) for cls_name, p in zip(classes_raw, proba)}

    # 5. Regola decisionale calibrata con soglia ottimale theta_high
    theta_high = float(opt_cfg.get("theta_high", 0.44))
    p_high = proba_map.get("high", 0.0)
    if p_high >= theta_high:
        pred_label = "high"
    else:
        non_high = {k: v for k, v in proba_map.items() if k != "high"}
        pred_label = max(non_high, key=non_high.get) if non_high else classes_raw[int(np.argmax(proba))]

    # 6. Indice di incertezza (Entropia di Shannon)
    entropy = compute_shannon_entropy(proba)
    entropy_thresh = float(opt_cfg.get("entropy_threshold", 1.15))
    needs_review = entropy >= entropy_thresh

    return {
        "priority": pred_label,
        "proba": proba_map,
        "entropy": entropy,
        "needs_review": needs_review,
        "cleaned_text": cleaned_text,
        "theta_high": theta_high,
    }