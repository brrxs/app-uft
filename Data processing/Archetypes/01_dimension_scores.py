# =============================================================================
# Archetypes/01_dimension_scores.py — 9 curated dimension scores, IES×APDA grain
# =============================================================================
# Reads Archetypes/config.yaml (dimensions, target, filters) and
# Archetypes/data/ies_apda_features.csv (built by Archetypes/00_prepare_data.py).
# Computes the same NaN-aware weighted dimension scores Vector/01_vectors.py
# uses for its PCA biplots (shared logic: Utils/data_prep.compute_dimension_scores),
# then applies the "Años acreditación ≥ acred_min_filter" filter so archetypal
# analysis is fit on the same institutions Vector's plots actually show.
#
# Usage (from project root):
#   python Archetypes/00_prepare_data.py
#   python Archetypes/01_dimension_scores.py
# =============================================================================

# %% Setup
import os
import sys
import yaml
import pandas as pd

sys.path.insert(0, "Utils")
from data_prep import compute_dimension_scores

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# %% Config
# Los paths salen de argv para que otra rama del pipeline (ej. Cine/, que
# agrupa por area CINE-F 13 en vez de por APDA) reuse este script tal cual:
#   python Archetypes/01_dimension_scores.py <config.yaml> <out_dir>
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "Archetypes/config.yaml"
OUT_DIR     = sys.argv[2] if len(sys.argv) > 2 else "Archetypes/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

DATA_PATH        = cfg["data_source"]
NORM_METHOD      = cfg.get("normalization", "zscore_within_apda")
USE_WEIGHTS      = cfg.get("use_weights", True)
DIMENSIONS       = cfg["dimensions"]
if not USE_WEIGHTS:
    DIMENSIONS = {
        name: {**d, "variables": {v: w for v, w in d["variables"].items()
                                   if not v.startswith("w_")}}
        for name, d in DIMENSIONS.items()
    }
TARGET_COL       = cfg["target"]
ACRED_MIN_FILTER = cfg["acred_min_filter"]
id_cols          = cfg["id_cols"]

# %% Load data
print(f"Loading: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"\nError: data file not found: {DATA_PATH}")
    print("Corre el 00_prepare_data.py de esta rama primero.")
    sys.exit(1)

agg = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"  {agg.shape[0]} IES×APDA cells × {agg.shape[1]} columns")

# %% Dimension scores (shared math with Vector/01_vectors.py)
dim_scores, valid_dims = compute_dimension_scores(
    agg, DIMENSIONS, id_cols, target_col=TARGET_COL, norm_method=NORM_METHOD,
    direction=cfg.get("direction", {}),
)
print(f"\nAvailable dimensions ({len(valid_dims)}): {valid_dims}")

# %% Filter to acred_min_filter (matches what Vector's plots show — see config.yaml)
_before = len(dim_scores)
dim_scores = dim_scores[
    pd.to_numeric(dim_scores[TARGET_COL], errors="coerce") >= ACRED_MIN_FILTER
].reset_index(drop=True)
print(f"\nFilter ≥{ACRED_MIN_FILTER} años acreditación: {_before} → {len(dim_scores)} IES×APDA cells")
print(dim_scores.groupby("apda").size().to_string())

# %% Save
out_path = f"{OUT_DIR}/dimension_scores.csv"
dim_scores.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nSaved: {out_path}  ({len(dim_scores)} rows, {len(valid_dims)} dimensions)")
