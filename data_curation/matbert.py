"""
MatBERT Pipeline for Thermoelectric Materials (Fixed)
=====================================================
Fixes:
  1. Correct MatBERT model path (m3rg-iitd/matbert)
  2. Formula → descriptive text conversion for meaningful embeddings
     e.g. "Bi2Te3" → "bismuth telluride Bi 0.40 Te 0.60"
  3. Mean pooling instead of CLS token for better embeddings

Steps:
  1. Generate MatBERT embeddings for each unique composition
  2. Compute cosine similarity (BERT-based) for a query material
  3. Use embeddings as features for ZT prediction (XGBoost / RF)

Requirements:
    pip install transformers torch scikit-learn xgboost pandas numpy
"""

import re
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Element name lookup (symbol → full name)
# ─────────────────────────────────────────────

ELEMENT_NAMES = {
    "H":"hydrogen","He":"helium","Li":"lithium","Be":"beryllium","B":"boron",
    "C":"carbon","N":"nitrogen","O":"oxygen","F":"fluorine","Ne":"neon",
    "Na":"sodium","Mg":"magnesium","Al":"aluminum","Si":"silicon","P":"phosphorus",
    "S":"sulfur","Cl":"chlorine","Ar":"argon","K":"potassium","Ca":"calcium",
    "Sc":"scandium","Ti":"titanium","V":"vanadium","Cr":"chromium","Mn":"manganese",
    "Fe":"iron","Co":"cobalt","Ni":"nickel","Cu":"copper","Zn":"zinc",
    "Ga":"gallium","Ge":"germanium","As":"arsenic","Se":"selenium","Br":"bromine",
    "Kr":"krypton","Rb":"rubidium","Sr":"strontium","Y":"yttrium","Zr":"zirconium",
    "Nb":"niobium","Mo":"molybdenum","Tc":"technetium","Ru":"ruthenium","Rh":"rhodium",
    "Pd":"palladium","Ag":"silver","Cd":"cadmium","In":"indium","Sn":"tin",
    "Sb":"antimony","Te":"tellurium","I":"iodine","Xe":"xenon","Cs":"cesium",
    "Ba":"barium","La":"lanthanum","Ce":"cerium","Pr":"praseodymium","Nd":"neodymium",
    "Pm":"promethium","Sm":"samarium","Eu":"europium","Gd":"gadolinium","Tb":"terbium",
    "Dy":"dysprosium","Ho":"holmium","Er":"erbium","Tm":"thulium","Yb":"ytterbium",
    "Lu":"lutetium","Hf":"hafnium","Ta":"tantalum","W":"tungsten","Re":"rhenium",
    "Os":"osmium","Ir":"iridium","Pt":"platinum","Au":"gold","Hg":"mercury",
    "Tl":"thallium","Pb":"lead","Bi":"bismuth","Po":"polonium","At":"astatine",
    "Rn":"radon","Fr":"francium","Ra":"radium","Ac":"actinium","Th":"thorium",
    "Pa":"protactinium","U":"uranium","Np":"neptunium","Pu":"plutonium",
    "Am":"americium","Cm":"curium","Co":"cobalt","Cf":"californium"
}


# ─────────────────────────────────────────────
# 1. Formula Parser & Text Converter
# ─────────────────────────────────────────────

def parse_formula(formula: str) -> dict:
    """Parse formula string → {element: fraction}"""
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    composition = defaultdict(float)
    for element, count in matches:
        composition[element] += float(count) if count else 1.0
    total = sum(composition.values())
    if total == 0:
        raise ValueError(f"Could not parse formula: {formula}")
    return {el: amt / total for el, amt in composition.items()}


def formula_to_text(formula: str) -> str:
    """
    Convert formula to descriptive text for BERT input.
    'Bi2Te3' → 'bismuth bismuth telluride telluride telluride Bi 0.40 Te 0.60'

    This gives BERT meaningful tokens instead of treating the
    formula as an unknown string.
    """
    try:
        comp = parse_formula(formula)
    except Exception:
        return formula  # fallback: use raw formula

    parts = []
    # Add element names (repeated by relative count for emphasis)
    for el, frac in sorted(comp.items(), key=lambda x: -x[1]):
        name = ELEMENT_NAMES.get(el, el.lower())
        parts.append(name)

    # Add element symbols with fractions
    for el, frac in sorted(comp.items(), key=lambda x: -x[1]):
        parts.append(f"{el} {frac:.2f}")

    return " ".join(parts)


# ─────────────────────────────────────────────
# 2. Load MatBERT
# ─────────────────────────────────────────────

def load_matbert(device):
    """
    Load MatBERT. Tries multiple known HuggingFace paths.
    Falls back to SciBERT if none found.
    """
    print("Loading MatBERT model...")

    candidates = [
        "m3rg-iitd/matbert",                    # IIT Delhi MatBERT
        "matsci-nlp/matbert-base-uncased",        # matsci-nlp
        "allenai/scibert_scivocab_uncased",       # SciBERT fallback
    ]

    for model_id in candidates:
        try:
            print(f"  Trying: {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model     = AutoModel.from_pretrained(model_id)
            print(f"  Loaded: {model_id}")
            break
        except Exception as e:
            print(f"  Failed : {e}")
            continue

    model = model.to(device)
    model.eval()
    print(f"  Running on: {device}\n")
    return tokenizer, model


# ─────────────────────────────────────────────
# 3. Generate Embeddings (Mean Pooling)
# ─────────────────────────────────────────────

def mean_pool(model_output, attention_mask) -> np.ndarray:
    """
    Mean pooling over token embeddings (better than CLS for similarity tasks).
    """
    token_embeddings = model_output.last_hidden_state
    mask_expanded    = attention_mask.unsqueeze(-1).float()
    sum_embeddings   = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask         = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return (sum_embeddings / sum_mask).cpu().numpy()


def generate_embeddings(formulas: list, tokenizer, model, device,
                        batch_size: int = 32) -> np.ndarray:
    """
    Generate mean-pooled embeddings for a list of formula strings.
    Converts formulas to descriptive text before embedding.
    """
    # Convert formulas → descriptive text
    texts = [formula_to_text(f) for f in formulas]

    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = mean_pool(outputs, inputs["attention_mask"])
        all_embeddings.append(embeddings)

        done = min(i + batch_size, total)
        if done % 320 == 0 or done == total:
            print(f"  Embedded {done}/{total} compositions")

    return np.vstack(all_embeddings)


# ─────────────────────────────────────────────
# 4. Cosine Similarity
# ─────────────────────────────────────────────

def cosine_similarity_vec(vec_a: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vec_a and every row in matrix."""
    norms   = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm_a  = np.linalg.norm(vec_a)
    safe    = np.where(norms == 0, 1e-9, norms)
    return (matrix @ vec_a) / (safe.squeeze() * (norm_a + 1e-9))


def find_most_similar_bert(query_formula: str,
                            deduped_df: pd.DataFrame,
                            embeddings: np.ndarray,
                            tokenizer, model, device,
                            formula_col: str = "composition",
                            top_n: int = 10) -> pd.DataFrame:
    """Find top_n most similar compositions using BERT cosine similarity."""
    query_text = formula_to_text(query_formula)
    inputs = tokenizer(
        query_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    query_emb   = mean_pool(outputs, inputs["attention_mask"]).squeeze()
    similarities = cosine_similarity_vec(query_emb, embeddings)

    result_df = deduped_df.copy()
    result_df["bert_cosine_similarity"] = similarities
    result_df = result_df.sort_values("bert_cosine_similarity", ascending=False)

    return result_df.head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────
# 5. ZT Prediction (XGBoost / RF)
# ─────────────────────────────────────────────

def train_and_evaluate(X: np.ndarray, y: np.ndarray,
                       model_type: str = "xgboost") -> dict:
    """Train ZT prediction model using BERT embeddings as features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if model_type == "xgboost":
        reg = XGBRegressor(
            n_estimators     = 500,
            learning_rate    = 0.05,
            max_depth        = 6,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = 42,
            tree_method      = "hist",
            device           = "cuda",
            verbosity        = 0
        )
    else:
        reg = RandomForestRegressor(
            n_estimators = 300,
            random_state = 42,
            n_jobs       = -1
        )

    print(f"Training {model_type.upper()}...")
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")

    return {"model": reg, "scaler": scaler,
            "y_test": y_test, "y_pred": y_pred,
            "mae": mae, "r2": r2}


def predict_zt(formula: str, trained_model, scaler,
               tokenizer, model, device) -> float:
    """Predict ZT for a new composition."""
    text   = formula_to_text(formula)
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=128,
                       padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    emb   = mean_pool(outputs, inputs["attention_mask"])
    emb_s = scaler.transform(emb)
    return float(trained_model.predict(emb_s)[0])


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Settings ───────────────────────────────────────────────────
    CSV_PATH    = "/home/sudarshan/Documents/myturn/datasets/final_featured_ID_dataset.csv"   # ← your CSV file
    FORMULA_COL = "composition"        # ← formula column
    ZT_COL      = "ZT"                 # ← ZT column
    QUERY       = "Bi2Te3"             # ← query material (change freely!)
    TOP_N       = 10
    MODEL_TYPE  = "xgboost"            # ← "xgboost" or "random_forest"

    # ── Device ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load MatBERT ───────────────────────────────────────────────
    tokenizer, bert_model = load_matbert(device)

    # ── Load & deduplicate dataset ─────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded dataset: {len(df)} rows")

    deduped_df = (
        df.groupby(FORMULA_COL)[ZT_COL]
          .agg(avg_ZT="mean", num_measurements="count")
          .reset_index()
    )
    print(f"Unique compositions: {len(deduped_df)}\n")

    formulas  = deduped_df[FORMULA_COL].tolist()
    zt_values = deduped_df["avg_ZT"].values

    # ── Step 1: Generate embeddings ────────────────────────────────
    print(f"Generating embeddings for {len(formulas)} compositions...")
    embeddings = generate_embeddings(formulas, tokenizer, bert_model, device)
    print(f"Embedding matrix shape: {embeddings.shape}")
    np.save("matbert_embeddings.npy", embeddings)
    print("Saved → matbert_embeddings.npy\n")

    # ── Step 2: BERT cosine similarity ────────────────────────────
    print(f"Query: {QUERY}  →  text: '{formula_to_text(QUERY)}'")
    print(f"\nTop {TOP_N} most similar (BERT cosine similarity):\n")
    sim_results = find_most_similar_bert(
        QUERY, deduped_df, embeddings,
        tokenizer, bert_model, device,
        formula_col=FORMULA_COL, top_n=TOP_N
    )
    print(sim_results[[FORMULA_COL, "avg_ZT",
                        "num_measurements",
                        "bert_cosine_similarity"]].to_string(index=False))
    sim_results.to_csv("bert_similar_materials.csv", index=False)

    # ── Step 3: ZT prediction ─────────────────────────────────────
    print(f"\n── ZT Prediction ──")
    results = train_and_evaluate(embeddings, zt_values, model_type=MODEL_TYPE)

    print(f"\nPredicted avg ZT for '{QUERY}': "
          f"{predict_zt(QUERY, results['model'], results['scaler'], tokenizer, bert_model, device):.4f}")

    print("\nDone! ✓")
