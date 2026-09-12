# =============================================================================
# Vector/01_vectors.py — Dimension scores + manual scatter + PCA biplot
# =============================================================================
# Reads Vector/config.yaml (dimensions, axes, target, filters, PCA/plot params)
# and Vector/data/ies_apda_features.csv (built by Vector/00_prepare_data.py).
#
# Usage (from project root):
#   python Vector/00_prepare_data.py
#   python Vector/01_vectors.py
# =============================================================================

# %% Setup
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import yaml

sys.path.insert(0, "Utils")
from utils import abreviaciones_ies
from data_prep import compute_dimension_scores, build_axis as _build_axis

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# %% Config
CONFIG_PATH = "Vector/config.yaml"
OUT_DIR     = "Vector/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
        cfg = yaml.safe_load(_f)
except FileNotFoundError:
    print(f"Config not found: {CONFIG_PATH}")
    print("Run Vector/build_config_template.py first to generate it.")
    sys.exit(1)

DATA_PATH        = cfg["data_source"]
NORM_METHOD      = cfg.get("normalization", "zscore_within_apda")
USE_WEIGHTS      = cfg.get("use_weights", True)
DIMENSIONS       = cfg["dimensions"]
if not USE_WEIGHTS:
    # w_* variables stay out of scoring; they still pass through as raw
    # columns in dimension_scores.csv (see w_cols block below).
    DIMENSIONS = {
        name: {**d, "variables": {v: w for v, w in d["variables"].items()
                                   if not v.startswith("w_")}}
        for name, d in DIMENSIONS.items()
    }
ANNOTATE         = cfg.get("pca", {}).get("annotate_points", True)
WATERMARK        = cfg.get("watermark", False)
TARGET_COL       = cfg["target"]
ACRED_MIN_FILTER = cfg["acred_min_filter"]
id_cols          = cfg["id_cols"]
AXIS_X           = cfg["axes"]["x"]
AXIS_Y           = cfg["axes"]["y"]
AXES_BY_APDA     = cfg.get("axes_by_apda", {})
AXIS_X_BY_APDA   = {a: c["x"] for a, c in AXES_BY_APDA.items() if "x" in c}
AXIS_Y_BY_APDA   = {a: c["y"] for a, c in AXES_BY_APDA.items() if "y" in c}
AXIS_LABELS      = cfg.get("axis_labels", {})
PCA_CFG          = cfg["pca"]
PLOT_CFG         = cfg["plot"]

# %% Load data
print(f"Loading: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"\nError: data file not found: {DATA_PATH}")
    print("Run Vector/00_prepare_data.py first to generate the IES×APDA feature matrix.")
    sys.exit(1)

agg = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"  {agg.shape[0]} IES×APDA cells × {agg.shape[1]} columns")

# w_* columns: always kept as raw pass-through columns in the output CSV,
# regardless of whether use_weights scores them into a dimension above.
w_cols = [c for c in agg.columns if c.startswith("w_")]
if w_cols:
    print(f"  w_* columns (raw pass-through): {w_cols}")

# %% Normalize variables to 0-100 % and compute dimension scores
# (shared with any other branch that consumes an IES×APDA table — see
# Utils/data_prep.py::compute_dimension_scores)
dim_scores, valid_dims = compute_dimension_scores(
    agg, DIMENSIONS, id_cols, target_col=TARGET_COL, norm_method=NORM_METHOD,
    direction=cfg.get("direction", {}),
)

# Pass-through columns
for wc in w_cols:
    dim_scores[wc] = agg[wc].values

dim_scores.to_csv(f"{OUT_DIR}/dimension_scores.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUT_DIR}/dimension_scores.csv  "
      f"({len(dim_scores)} rows, {len(valid_dims)} dimensions)")

# Abbreviated IES labels for plots (falls back to ies_norm when not in dict)
dim_scores["ies_label"] = (
    dim_scores["ies_norm"].map(abreviaciones_ies).fillna(dim_scores["ies_norm"])
)

# Hard filter: keep only IES with >= ACRED_MIN_FILTER years of accreditation
if TARGET_COL in dim_scores.columns:
    _before = len(dim_scores)
    dim_scores = dim_scores[
        pd.to_numeric(dim_scores[TARGET_COL], errors="coerce") >= ACRED_MIN_FILTER
    ].reset_index(drop=True)
    print(f"\nFilter ≥{ACRED_MIN_FILTER} años acreditación: {_before} → {len(dim_scores)} IES×APDA cells")

# %% Plot colors — accreditation years + UFT highlight
# Edit ACRED_YEAR_COLORS to customize the palette (keys are años de acreditación,
# 1-7 per CNA Chile). Any year not listed here falls back to tab10.
ACRED_YEAR_COLORS = {
    1: "#4C72B0",
    2: "#DD8452",
    3: "#55A868",
    4: "#4C72B0",
    5: "#DD8452",
    6: "#55A868",
    7: "#8172B2",
}
UFT_IES_NORM = "universidad finis terrae"
UFT_COLOR    = "#00b7eb"

# Qualitative color mapping by accreditation years (consistent across all APDA plots)
from matplotlib.patches import Patch
CMAP_COL = TARGET_COL if TARGET_COL in dim_scores.columns else None
if CMAP_COL:
    _year_vals     = sorted(pd.to_numeric(dim_scores[CMAP_COL], errors="coerce")
                            .dropna().astype(int).unique())
    _fallback      = plt.cm.get_cmap("tab10", len(_year_vals))
    COLOR_MAP_QUAL = {v: ACRED_YEAR_COLORS.get(v, _fallback(i))
                       for i, v in enumerate(_year_vals)}
    print(f"Color categories (años acreditación): {_year_vals}")
else:
    COLOR_MAP_QUAL = {}
    print(f"Warning: '{TARGET_COL}' not found — points will be uncolored.")

# UFT always shown in its brand color, regardless of accreditation-year category
dim_scores["_is_uft"] = dim_scores["ies_norm"].eq(UFT_IES_NORM)
if dim_scores["_is_uft"].any():
    print(f"UFT rows highlighted in {UFT_COLOR}: {dim_scores['_is_uft'].sum()}")

print(f"\nAvailable dimensions: {valid_dims}")

# =============================================================================
# %% Axis definition
# =============================================================================
# AXIS_X / AXIS_Y come from `axes:` in Vector/config.yaml — edit them there
# (keys must match dimension names above, printed as "Available dimensions").

# %% Helpers
def _add_watermark(ax):
    ax.text(0.5, 0.5, "PRELIMINAR - NO REPRODUCIR",
            transform=ax.transAxes, fontsize=22, color="gray",
            alpha=0.18, ha="center", va="center", rotation=30,
            fontweight="bold",
            bbox=dict(boxstyle="round", fc="none", ec="none"))


# _build_axis is now the shared Utils/data_prep.py::build_axis (imported above
# as _build_axis) — was duplicated verbatim here and in Dashboard/build_dashboard.py.

# %% Manual scatter — one figure per APDA
x_vals, x_label = _build_axis(AXIS_X, dim_scores, valid_dims,
                               apda_col="apda", axes_by_apda=AXIS_X_BY_APDA or None)
y_vals, y_label = _build_axis(AXIS_Y, dim_scores, valid_dims,
                               apda_col="apda", axes_by_apda=AXIS_Y_BY_APDA or None)

if x_vals is None or y_vals is None:
    print(f"\nSkipping manual plots: one or both axes have no valid dimensions.")
    print(f"Available dimensions: {valid_dims}")
else:
    apdas = sorted(dim_scores["apda"].dropna().unique())
    print(f"\nGenerating {len(apdas)} APDA scatter plots...")
    print(f"  X: {x_label}")
    print(f"  Y: {y_label}")

    # Composite axes are weighted means of several already-bounded 0-100 dimension
    # scores, which shrinks their spread toward the middle (averaging several
    # roughly-independent bounded variables cancels out real differences). Convert
    # to a percentile rank within each APDA peer group so institutions spread across
    # the full 0-100 canvas based on relative standing among peers, instead of
    # being squeezed by that compounded averaging.
    plot_df = dim_scores.reset_index(drop=True).copy()
    plot_df["_x_raw"] = x_vals.values
    plot_df["_y_raw"] = y_vals.values
    plot_df["_x"] = plot_df.groupby("apda")["_x_raw"].rank(pct=True) * 100.0
    plot_df["_y"] = plot_df.groupby("apda")["_y_raw"].rank(pct=True) * 100.0

    for apda in apdas:
        sub = plot_df[plot_df["apda"] == apda]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=tuple(PLOT_CFG["figsize"]))

        if CMAP_COL:
            _yrs = pd.to_numeric(sub[CMAP_COL], errors="coerce")
            pt_colors = [COLOR_MAP_QUAL.get(int(v), "grey") if pd.notna(v) else "grey"
                         for v in _yrs]
        else:
            pt_colors = ["steelblue"] * len(sub)
        pt_colors = [UFT_COLOR if is_uft else c
                     for is_uft, c in zip(sub["_is_uft"], pt_colors)]
        ax.scatter(sub["_x"], sub["_y"],
                   color=pt_colors,
                   s=90, alpha=0.9,
                   edgecolors="white", linewidths=0.6, zorder=3)
        if CMAP_COL:
            _present = sorted(_yrs.dropna().astype(int).unique())
            handles  = [Patch(color=COLOR_MAP_QUAL[v], label=f"{v} años")
                        for v in _present if v in COLOR_MAP_QUAL]
            if sub["_is_uft"].any():
                handles.append(Patch(color=UFT_COLOR, label="UFT"))
            ax.legend(handles=handles, title="Años acreditación",
                      fontsize=8, title_fontsize=8, loc="upper left")

        for _, row in sub.iterrows():
            if pd.notna(row["_x"]) and pd.notna(row["_y"]):
                ax.annotate(row["ies_label"], (row["_x"], row["_y"]),
                            fontsize=8, alpha=0.9,
                            textcoords="offset points", xytext=(5, 4))

        ax.set_xlabel(AXIS_LABELS.get("x", x_label) + "  (percentil dentro de la APDA)", fontsize=9)
        ax.set_ylabel(AXIS_LABELS.get("y", y_label) + "  (percentil dentro de la APDA)", fontsize=9)
        ax.set_title(f"APDA: {apda}", fontsize=13, pad=12)
        _axis_lo, _axis_hi = PLOT_CFG["axis_limits"]
        ax.set_xlim(_axis_lo, _axis_hi)
        ax.set_ylim(_axis_lo, _axis_hi)
        ax.axhline(50, color="#cccccc", linestyle="--", linewidth=0.8, zorder=1)
        ax.axvline(50, color="#cccccc", linestyle="--", linewidth=0.8, zorder=1)
        ax.grid(True, alpha=0.25, zorder=0)
        if WATERMARK:
            _add_watermark(ax)
        plt.tight_layout()
        plt.show()

# %% PCA biplot — one figure per APDA
def _pca_plot(sub_df, apda, valid_dims, annotate=True,
              cmap_col=None, color_map=None):
    """Fit PCA on sub_df rows and save biplot for one APDA."""
    # a dimension can be entirely NaN within this APDA's rows (e.g. a
    # dimension scoped to a single other APDA, like "Acreditación
    # Trayectoria" outside Salud y bienestar); SimpleImputer silently drops
    # such columns, so exclude them upfront to keep valid_dims aligned with
    # pca.components_'s column count.
    valid_dims = [d for d in valid_dims if sub_df[d].notna().any()]
    X_mat = sub_df[valid_dims].copy()
    n_comp = min(PCA_CFG["n_components"], X_mat.shape[0] - 1, X_mat.shape[1])
    if n_comp < PCA_CFG["n_components"]:
        print(f"  [{apda}] Too few institutions for PCA ({X_mat.shape[0]}), skipping.")
        return None

    X_imp   = SimpleImputer(strategy=PCA_CFG["impute_strategy"]).fit_transform(X_mat)
    X_std   = StandardScaler().fit_transform(X_imp)
    pca     = PCA(n_components=PCA_CFG["n_components"], random_state=PCA_CFG["random_state"])
    scores  = pca.fit_transform(X_std)
    var_exp = pca.explained_variance_ratio_

    load_max    = max(abs(pca.components_[0]).max(), abs(pca.components_[1]).max(), 1e-9)
    score_max   = max(abs(scores[:, 0]).max(), abs(scores[:, 1]).max(), 1e-9)
    arrow_scale = PCA_CFG["arrow_scale_factor"] * score_max / load_max

    if cmap_col and color_map:
        _yrs      = pd.to_numeric(sub_df[cmap_col], errors="coerce")
        pt_colors = [color_map.get(int(v), "grey") if pd.notna(v) else "grey"
                     for v in _yrs]
    else:
        pt_colors = ["steelblue"] * len(sub_df)
        _yrs      = None
    _is_uft   = sub_df["_is_uft"].values
    pt_colors = [UFT_COLOR if is_uft else c for is_uft, c in zip(_is_uft, pt_colors)]

    fig, ax = plt.subplots(figsize=tuple(PLOT_CFG["figsize"]))
    ax.scatter(scores[:, 0], scores[:, 1],
               color=pt_colors,
               s=90, alpha=0.9,
               edgecolors="white", linewidths=0.6, zorder=3)
    if cmap_col and color_map and _yrs is not None:
        _present = sorted(_yrs.dropna().astype(int).unique())
        handles  = [Patch(color=color_map[v], label=f"{v} años")
                    for v in _present if v in color_map]
        if _is_uft.any():
            handles.append(Patch(color=UFT_COLOR, label="UFT"))
        ax.legend(handles=handles, title="Años acreditación",
                  fontsize=8, title_fontsize=8, loc="upper left")

    if annotate:
        for i, ies in enumerate(sub_df["ies_label"].values):
            ax.annotate(ies, (scores[i, 0], scores[i, 1]),
                        fontsize=8, alpha=0.9,
                        textcoords="offset points", xytext=(5, 4))

    for j, dim_name in enumerate(valid_dims):
        ax_end = pca.components_[0, j] * arrow_scale
        ay_end = pca.components_[1, j] * arrow_scale
        ax.annotate("", xy=(ax_end, ay_end), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.8, zorder=5))
        ax.text(ax_end * 1.12, ay_end * 1.12, dim_name,
                fontsize=8, color="crimson", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                zorder=6)

    ax.axhline(0, color="#cccccc", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"PC1  ({var_exp[0]*100:.1f} % varianza explicada)", fontsize=11)
    ax.set_ylabel(f"PC2  ({var_exp[1]*100:.1f} % varianza explicada)", fontsize=11)
    ax.set_title(f"PCA — APDA: {apda}", fontsize=13, pad=12)
    ax.grid(True, alpha=0.25, zorder=0)
    if WATERMARK:
        _add_watermark(ax)
    plt.tight_layout()
    plt.show()
    return pca.components_[:2], var_exp[:2], valid_dims


pca_loadings_records = []
if len(valid_dims) < 2:
    print("\nSkipping PCA: need at least 2 dimensions.")
else:
    apdas = sorted(dim_scores["apda"].dropna().unique())
    print(f"\nGenerating {len(apdas)} APDA PCA biplots...")
    pca_df = dim_scores.reset_index(drop=True)

    for apda in apdas:
        sub = pca_df[pca_df["apda"] == apda]
        result = _pca_plot(sub, apda, valid_dims, annotate=ANNOTATE,
                            cmap_col=CMAP_COL, color_map=COLOR_MAP_QUAL)
        if result is not None:
            components, var_exp, local_dims = result
            for j, dim_name in enumerate(local_dims):
                pca_loadings_records.append({
                    "apda": apda,
                    "dimension": dim_name,
                    "PC1": components[0, j],
                    "PC2": components[1, j],
                    "PC1_var_exp": var_exp[0],
                    "PC2_var_exp": var_exp[1],
                })

# %% PCA composition — PC1/PC2 loadings across all APDAs
# Each APDA fits its own PCA, so "PC1"/"PC2" are not directly comparable
# numbers across APDAs — this shows, per APDA, which dimensions drive each
# component (sign and magnitude of the loading), side by side.
if not pca_loadings_records:
    print("\nSkipping PCA composition summary: no APDA had a valid PCA.")
else:
    loadings_df = pd.DataFrame(pca_loadings_records)
    loadings_df.to_csv(f"{OUT_DIR}/pca_loadings_by_apda.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_DIR}/pca_loadings_by_apda.csv  ({len(loadings_df)} rows)")

    for pc in ("PC1", "PC2"):
        pivot = loadings_df.pivot(index="dimension", columns="apda", values=pc).reindex(valid_dims)
        var_by_apda = loadings_df.drop_duplicates("apda").set_index("apda")[f"{pc}_var_exp"]
        col_labels = [f"{a}\n({var_by_apda[a]*100:.0f}%)" for a in pivot.columns]

        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(pivot.columns) + 3),
                                         max(4, 0.5 * len(valid_dims) + 1.5)))
        vmax = np.nanmax(np.abs(pivot.values))
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > vmax * 0.6 else "black")
        ax.set_title(f"Composición de {pc} por dimensión, todas las APDAs\n"
                     f"(% bajo cada APDA = varianza explicada por {pc})", fontsize=12, pad=12)
        fig.colorbar(im, ax=ax, label="Loading")
        plt.tight_layout()
        plt.show()

# %% PCA composition comparison — top dimensions & cross-APDA similarity
# PCA component sign is arbitrary per fit (flipping every loading in a
# component is an equally valid solution), so "same variables" is judged by
# loading magnitude/rank here, and cross-APDA similarity uses |cosine
# similarity| so an arbitrary sign flip doesn't register as a difference.
if not pca_loadings_records:
    print("\nSkipping PCA comparison: no APDA had a valid PCA.")
else:
    TOP_N = 3
    print(f"\nTop {TOP_N} dimensions by |loading|, per APDA:")
    top_records = []
    for pc in ("PC1", "PC2"):
        print(f"\n  {pc}:")
        for apda in apdas:
            row = loadings_df[loadings_df["apda"] == apda].set_index("dimension")[pc]
            if row.empty:
                continue
            top = row.reindex(row.abs().sort_values(ascending=False).index).head(TOP_N)
            print(f"    {apda}: " + ", ".join(f"{d} ({v:+.2f})" for d, v in top.items()))
            for rank, (d, v) in enumerate(top.items(), start=1):
                top_records.append({"PC": pc, "apda": apda, "rank": rank,
                                     "dimension": d, "loading": v})
    pd.DataFrame(top_records).to_csv(f"{OUT_DIR}/pca_top_dimensions.csv",
                                      index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_DIR}/pca_top_dimensions.csv")

    # Cross-APDA similarity: |cosine similarity| between loading vectors, per
    # component. 1.0 = same dimensions drive the component in both APDAs
    # (same direction, up to sign); 0.0 = unrelated compositions.
    for pc in ("PC1", "PC2"):
        pivot = loadings_df.pivot(index="dimension", columns="apda", values=pc).reindex(valid_dims)
        mat = np.nan_to_num(pivot.to_numpy(), nan=0.0)
        norms = np.linalg.norm(mat, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        unit = mat / norms
        sim = np.abs(unit.T @ unit)

        fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(apdas) + 2),
                                         max(4, 0.7 * len(apdas) + 1)))
        im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.columns)))
        ax.set_yticklabels(pivot.columns, fontsize=8)
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if sim[i, j] < 0.6 else "black")
        ax.set_title(f"Similitud entre APDAs — composición de {pc}\n"
                     f"(|cosine similarity| entre vectores de loadings; 1 = misma composición)",
                     fontsize=11, pad=12)
        fig.colorbar(im, ax=ax, label="|cosine similarity|")
        plt.tight_layout()
        plt.show()

