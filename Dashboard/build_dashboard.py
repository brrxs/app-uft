# =============================================================================
# Dashboard/build_dashboard.py — Interactive Plotly dashboard for Vector + AA
# =============================================================================
# Read-only visualization layer. Reads ONLY the CSV/YAML outputs already
# written by Vector/00-01 and Archetypes/01-02 — never imports or re-runs any
# Vector/ or Archetypes/ script (those have top-level side effects: config
# loading, plt.show(), etc.). Writes a single self-contained HTML file to
# Dashboard/outputs/index.html, plus (untracked) one standalone page per APDA
# under Dashboard/BI testing/ — no header/tabs, no PCA views — meant to be
# embedded as an iframe inside a Power BI report.
#
# Per APDA, the institution map toggles between up to four views (index.html
# only; Dashboard/BI testing/ pages drop the last two):
#   - "Mapa vectorial": Vector/01_vectors.py's manual composite-axis scatter
#     (axes named via Vector/config.yaml's axis_labels, percentile-ranked within APDA)
#   - "PCA": Vector/01_vectors.py's plain PCA biplot (institutions only)
#   - "PCA + AA": a PCA map with archetype corners overlaid (median-impute ->
#     StandardScaler -> PCA(2) on the same standardized dimension scores AA
#     uses) without touching or refitting the AA model itself; archetype
#     profiles/alphas are read verbatim from the saved CSVs
#
# Usage (from this folder, APP UFT/Dashboard/):
#   python build_dashboard.py                 # rama APDA (default)
#   python build_dashboard.py Cine/config.yaml  # otra rama
#
# El argumento opcional es el config.yaml de otra rama del pipeline. Su bloque
# "dashboard:" pisa los paths, etiquetas y textos del default de mas abajo, y
# el resto del archivo (dimensions/axes/pca) reemplaza a Vector/config.yaml.
# Sin argumento el script produce exactamente el mismo HTML de siempre.
# =============================================================================

# %% Setup
import os
import sys
import io
import json
import uuid
import contextlib
from datetime import date

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, QhullError

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

sys.path.insert(0, "Utils")
from utils import abreviaciones_ies, slug
from data_prep import normalize_variables_0_100, compute_dimension_scores, build_axis as _shared_build_axis
from variable_labels import display_name, label_map, info_icon_html, ORIENT_LABEL
from report_html import load_text_blocks, md_to_html, DASHBOARD_CSS

# %% Branch descriptor
# Default = la rama APDA. Otra rama pisa lo que necesite via el bloque
# "dashboard:" de su propio config.yaml (ver Cine/config.yaml).
#
# parent_groups: cuando esta presente, la barra de pestanas pasa a dos niveles
#   ({parent_label} arriba, {group_label} abajo). Es un dict ordenado
#   {grupo padre: [grupo hijo, ...]} y tambien fija el orden de las pestanas.
#   Sin el, la barra queda de un solo nivel, como en la rama APDA.
BRANCH_DEFAULT = {
    "config":            "Vector/config.yaml",
    "aa_config":         "Archetypes/config.yaml",
    "features":          "Vector/data/ies_apda_features.csv",
    "vector_dim_scores": "Vector/outputs/dimension_scores.csv",
    "arch_dim_scores":   "Archetypes/outputs/dimension_scores.csv",
    "arch_memberships":  "Archetypes/outputs/archetype_memberships.csv",
    "arch_profile_tmpl": "Archetypes/outputs/archetype_profiles_{safe}.csv",
    "texto":             "build_dashboard_texto.md",
    "out_dir":           "outputs",
    "site_dir":          "site/apda",
    # Archetypes/02_archetypes.py eligio k=3 por codo en estas dos; aca se
    # muestran a k=4, refitteando AA en memoria (ver mas abajo). La rama que
    # ya guarda k=4 en sus outputs deja este dict vacio.
    "k_override":        {"Artes, humanidades y cultura": 4,
                          "Ciencias, tecnología y calidad de vida": 4},
    "parent_groups":     None,
    "parent_label":      "APDA",
    "title":             "Dashboard — Avances {date} Proyecto APDAs",
    "heading":           "Dashboard — Avances {date}",
    "intro_note":        "",
    # Formas del nombre del grupo que aparecen en el texto visible.
    "group_label":       "APDA",
    "group_plural":      "APDAs",
    "group_within":      "dentro de la APDA",
    "group_this":        "esta APDA",
    "group_each":        "cada APDA",
    "generate_bi_pages": True,
}

BRANCH = dict(BRANCH_DEFAULT)
if len(sys.argv) > 1:
    with open(sys.argv[1], "r", encoding="utf-8") as _f:
        BRANCH.update(yaml.safe_load(_f).get("dashboard") or {})
    BRANCH["config"] = sys.argv[1]

# %% Paths
VECTOR_CONFIG      = BRANCH["config"]
VECTOR_FEATURES    = BRANCH["features"]
VECTOR_DIM_SCORES  = BRANCH["vector_dim_scores"]
ARCH_DIM_SCORES    = BRANCH["arch_dim_scores"]
ARCH_MEMBERSHIPS   = BRANCH["arch_memberships"]
ARCH_PROFILE_TMPL  = BRANCH["arch_profile_tmpl"]
GLOSSARY_PATH      = "Utils/variable_glossary.yaml"
OUT_DIR            = BRANCH["out_dir"]
OUT_PATH           = f"{OUT_DIR}/index.html"
os.makedirs(OUT_DIR, exist_ok=True)

def safe_id(name):
    """Nombre de grupo -> sufijo usable en ids del DOM y en nombres de archivo."""
    return name.replace("/", "-").replace(" ", "_")


PARENT_GROUPS = BRANCH["parent_groups"]
PARENT_LABEL  = BRANCH["parent_label"]
GROUP_LABEL   = BRANCH["group_label"]
GROUP_PLURAL  = BRANCH["group_plural"]
GROUP_WITHIN  = BRANCH["group_within"]
GROUP_THIS    = BRANCH["group_this"]
GROUP_EACH    = BRANCH["group_each"]
_cap = lambda t: t[:1].upper() + t[1:]
GROUP_EACH_CAP  = _cap(GROUP_EACH)
GROUP_LABEL_CAP = _cap(GROUP_LABEL)

TARGET_COL = "Años acreditación (30 de octubre de 2025)"
ID_COLS    = ["ies_norm", "apda"]
UFT_IES_NORM = "universidad finis terrae"
UFT_COLOR    = "#00b7eb"
BAR_COLOR    = "#3881BC"

ACRED_YEAR_COLORS = {
    1: "#4C72B0", 2: "#DD8452", 3: "#55A868", 4: "#4C72B0",
    5: "#DD8452", 6: "#55A868", 7: "#8172B2",
}
FALLBACK_YEAR_COLORS = ["#7f7f7f", "#bcbd22", "#17becf", "#9467bd"]
ARCHETYPE_COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]

PCA_N_COMPONENTS    = 2
PCA_IMPUTE_STRATEGY = "median"
PCA_RANDOM_STATE    = 0
ARROW_SCALE_FACTOR  = 0.65
AXIS_LIMITS         = (-4, 104)

TETRA_VERTS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
TETRA_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# %% Load config + data
with open(VECTOR_CONFIG, "r", encoding="utf-8") as f:
    vec_cfg = yaml.safe_load(f)
ACRED_MIN_FILTER = vec_cfg["acred_min_filter"]
AXIS_X_CFG = vec_cfg["axes"]["x"]
AXIS_Y_CFG = vec_cfg["axes"]["y"]
AXES_BY_APDA = vec_cfg.get("axes_by_apda", {})
AXIS_X_BY_APDA = {a: c["x"] for a, c in AXES_BY_APDA.items() if "x" in c}
AXIS_Y_BY_APDA = {a: c["y"] for a, c in AXES_BY_APDA.items() if "y" in c}
AXIS_LABELS = vec_cfg.get("axis_labels", {})
WATERMARK = vec_cfg.get("watermark", False)


def resolve_axis_cfg(apda, axis_key):
    """Per-APDA axes_by_apda override, falling back to the global axes.x/axes.y."""
    by_apda = AXIS_X_BY_APDA if axis_key == "x" else AXIS_Y_BY_APDA
    default = AXIS_X_CFG if axis_key == "x" else AXIS_Y_CFG
    return by_apda.get(apda, default)

with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
    var_glossary = yaml.safe_load(f)

for p in (VECTOR_DIM_SCORES, ARCH_DIM_SCORES, ARCH_MEMBERSHIPS):
    if not os.path.exists(p):
        print(f"Not found: {p}\nRun the Vector/ and Archetypes/ pipelines first.")
        sys.exit(1)

arch_dim_scores = pd.read_csv(ARCH_DIM_SCORES, encoding="utf-8-sig")
memberships     = pd.read_csv(ARCH_MEMBERSHIPS, encoding="utf-8-sig")
vec_dim_scores  = pd.read_csv(VECTOR_DIM_SCORES, encoding="utf-8-sig")

valid_dims = [c for c in arch_dim_scores.columns if c not in ID_COLS + [TARGET_COL]]
print(f"Dimensions: {valid_dims}")

merged = memberships.merge(arch_dim_scores[ID_COLS + valid_dims], on=ID_COLS, how="left")
merged["ies_label"] = merged["ies_norm"].map(abreviaciones_ies).fillna(merged["ies_norm"])

# ── Dashboard-only k=4 override ─────────────────────────────────────────────
# Archetypes/02_archetypes.py's elbow heuristic picked k=3 for these two
# APDAs; the user wants k=4 shown here instead, without touching that script,
# its config, or its saved outputs (which other analyses, e.g. archetypoids,
# still read at k=3). Refits AA in-memory from the same dimension_scores.csv
# input, using the same preprocessing/AA settings as 02_archetypes.py.
DASHBOARD_K_OVERRIDE = BRANCH["k_override"]
k4_profiles = {}
if DASHBOARD_K_OVERRIDE:
    import archetypes_compat  # noqa: F401 — sklearn/numpy shims, see Archetypes/02_archetypes.py
    from archetypes import AA as _AA

    with open(BRANCH["aa_config"], "r", encoding="utf-8") as f:
        _aa_cfg = yaml.safe_load(f)["aa"]
    _zero_fill_dims = ["Investigación ANID (APDA)", "Producción Científica (APDA)"]

    for _apda, _k in DASHBOARD_K_OVERRIDE.items():
        _sub = arch_dim_scores[arch_dim_scores["apda"] == _apda].reset_index(drop=True)
        _local_dims = [d for d in valid_dims if _sub[d].notna().any()]
        _X_pre = _sub[_local_dims].copy()
        for d in _zero_fill_dims:
            if d in _X_pre.columns:
                _X_pre[d] = _X_pre[d].fillna(0.0)
        _X_imp = SimpleImputer(strategy="median").fit_transform(_X_pre)
        _scaler = StandardScaler().fit(_X_imp)
        _X_std = _scaler.transform(_X_imp)

        _aa = _AA(n_archetypes=_k, n_init=_aa_cfg["n_init"], max_iter=_aa_cfg["max_iter"],
                  tol=_aa_cfg["tol"], random_state=_aa_cfg["random_state"])
        _aa.fit(_X_std)

        _profiles = pd.DataFrame(_scaler.inverse_transform(_aa.archetypes_), columns=_local_dims)
        _profiles.insert(0, "archetype", [f"A{i+1}" for i in range(_k)])
        _profiles.insert(0, "apda", _apda)
        k4_profiles[_apda] = _profiles

        _alphas = _aa.alphas_
        _dominant = _alphas.argmax(axis=1)
        _rows = _sub.copy()
        for i in range(_k):
            _rows[f"alpha_{i+1}"] = _alphas[:, i]
        _rows["dominant_archetype"] = [f"A{i+1}" for i in _dominant]
        _rows["dominant_share"] = _alphas.max(axis=1)
        _rows["k"] = _k
        _rows["ies_label"] = _rows["ies_norm"].map(abreviaciones_ies).fillna(_rows["ies_norm"])
        _rows["_is_uft"] = _rows["ies_norm"].eq(UFT_IES_NORM)

        merged = pd.concat([merged[merged["apda"] != _apda], _rows], ignore_index=True)

vec_dim_scores = vec_dim_scores[
    pd.to_numeric(vec_dim_scores[TARGET_COL], errors="coerce") >= ACRED_MIN_FILTER
].reset_index(drop=True)
vec_dim_scores["ies_label"] = vec_dim_scores["ies_norm"].map(abreviaciones_ies).fillna(vec_dim_scores["ies_norm"])
vec_dim_scores["_is_uft"]   = vec_dim_scores["ies_norm"].eq(UFT_IES_NORM)

_present = set(merged["apda"].dropna())
if PARENT_GROUPS:
    # el orden (y el agrupamiento) lo fija el YAML, no el alfabeto
    apdas = [g for gs in PARENT_GROUPS.values() for g in gs if g in _present]
else:
    apdas = sorted(_present)

# ── Site nav (Part A) ────────────────────────────────────────────────────────
# Both branch runs need the full parent->children mapping: the APDA branch to
# link down to its CINE areas, the CINE branch to link back up to its APDA and
# across sibling areas. Always loaded from Cine/config.yaml (single source of
# truth for the grouping, see its dashboard.parent_groups) regardless of which
# branch is running.
if PARENT_GROUPS:
    SITE_PARENT_GROUPS = PARENT_GROUPS
else:
    with open("Cine/config.yaml", "r", encoding="utf-8") as _f:
        SITE_PARENT_GROUPS = (yaml.safe_load(_f).get("dashboard") or {}).get("parent_groups") or {}
SITE_AREA_TO_PARENT = {a: p for p, gs in SITE_PARENT_GROUPS.items() for a in gs}


def _site_nav(current_label, is_cine):
    """Breadcrumb nav for a standalone Dashboard/site/ page, reusing the
    .tabs/.tab-btn CSS as anchors instead of buttons. From an APDA page: links
    to its CINE areas. From a CINE page: link back to its parent APDA plus all
    sibling areas (current one shown active, not a link)."""
    top = [
        '<a class="tab-btn" href="../index.html">Inicio</a>',
        '<a class="tab-btn" href="../index.html#decisiones">Decisiones</a>',
    ]
    parent = SITE_AREA_TO_PARENT.get(current_label) if is_cine else current_label
    if is_cine and parent:
        top.append(f'<a class="tab-btn" href="../apda/{slug(parent)}.html">{parent}</a>')
    elif not is_cine:
        top.append(f'<span class="tab-btn active">{current_label}</span>')

    rows = []
    for child in SITE_PARENT_GROUPS.get(parent, []) if parent else []:
        active = is_cine and child == current_label
        href = f"{slug(child)}.html" if is_cine else f"../cine/{slug(child)}.html"
        rows.append(f'<span class="tab-btn active">{child}</span>' if active
                    else f'<a class="tab-btn" href="{href}">{child}</a>')
    second_row = (
        '<div class="tabs group-row" style="display:flex;">'
        '<span class="row-label">Áreas CINE-F 13 de esta APDA:</span>'
        + "".join(rows) + "</div>"
    ) if rows else ""
    return f'<div class="tabs">{"".join(top)}</div>{second_row}'


# ── Raw variables (for the axis -> dimension -> variable bar-chart selector) ──
vec_features = pd.read_csv(VECTOR_FEATURES, encoding="utf-8-sig")
vec_features = vec_features[
    pd.to_numeric(vec_features[TARGET_COL], errors="coerce") >= ACRED_MIN_FILTER
].reset_index(drop=True)
vec_features["ies_label"] = vec_features["ies_norm"].map(abreviaciones_ies).fillna(vec_features["ies_norm"])
vec_features["_is_uft"]   = vec_features["ies_norm"].eq(UFT_IES_NORM)

# apda -> axis -> {label, dims: {dimension_name: [variable, ...]}}, mirroring
# Vector/config.yaml — resolved per APDA (via resolve_axis_cfg) rather than off
# the flat global axes.x/axes.y, so a dimension that only exists in
# axes_by_apda (e.g. "Matrícula Especialidades", Salud y bienestar only)
# still shows up in that APDA's own "Variable individual por institución"
# selector instead of being invisible everywhere.
def _axis_dims(axis_cfg):
    dims = {}
    for dim_name in axis_cfg:
        if dim_name not in valid_dims:
            continue
        var_names = [v for v in vec_cfg["dimensions"][dim_name]["variables"] if v in vec_features.columns]
        if var_names:
            dims[dim_name] = var_names
    return dims

AXIS_DIM_VAR_BY_APDA = {
    safe_id(apda): {
        axis_key: {"label": AXIS_LABELS.get(axis_key, axis_key), "dims": _axis_dims(resolve_axis_cfg(apda, axis_key))}
        for axis_key in vec_cfg["axes"]
    }
    for apda in apdas
}

bar_all_vars = sorted({
    v for a in AXIS_DIM_VAR_BY_APDA.values() for ax in a.values() for vs in ax["dims"].values() for v in vs
})

# dimension -> color (stable per dimension, cycling a qualitative palette) and
# variable -> dimension, used by the "Mapa interactivo" results table to
# color-code each row like the archetype "magnitud de cambio" bar charts.
DIM_COLOR_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]
_dim_names_all = sorted({d for a in AXIS_DIM_VAR_BY_APDA.values() for ax in a.values() for d in ax["dims"]})
DIM_COLOR_JS = {d: DIM_COLOR_PALETTE[i % len(DIM_COLOR_PALETTE)] for i, d in enumerate(_dim_names_all)}
VAR_DIMENSION_JS = {
    v: d for a in AXIS_DIM_VAR_BY_APDA.values() for ax in a.values() for d, vs in ax["dims"].items() for v in vs
}

# variable -> showcase name (Utils/variable_glossary.yaml's nombre_corto),
# used by the JS variable selector/dropdown and chart titles so the user
# never sees a raw technical column name.
VAR_LABEL_JS = label_map(bar_all_vars, var_glossary)

# ── Glosario de variables (tooltips) — solo variables activas mostradas en el
# dashboard, para no inflar el HTML con la mitad "retirada" del glosario ─────
VAR_GLOSSARY_JS = {
    var: {
        "nombre": entry["nombre"],
        "descripcion": entry["descripcion"],
        "calculo": entry["calculo"],
        "fuente": entry["fuente"],
        "unidad": entry["unidad"],
        "orientacion_label": ORIENT_LABEL.get(entry["orientacion"], entry["orientacion"]),
    }
    for var, entry in var_glossary.items()
    if var in bar_all_vars and entry.get("estado") == "activa"
}
missing_glossary = [v for v in bar_all_vars if v not in VAR_GLOSSARY_JS]
if missing_glossary:
    print(f"Warning: {len(missing_glossary)} variable(s) sin entrada en el glosario: {missing_glossary}")

bar_norm_cols = normalize_variables_0_100(
    vec_features, bar_all_vars, direction=vec_cfg.get("direction", {}), apda_col="apda", verbose=False,
)
norm_df = pd.DataFrame({f"{var}__norm": s for var, s in bar_norm_cols.items()})
vec_features = pd.concat([vec_features, norm_df], axis=1)

bar_default_var = next(
    (vs[0] for a in AXIS_DIM_VAR_BY_APDA.values() for ax in a.values() for vs in ax["dims"].values() if vs),
    None,
)
print(f"{GROUP_PLURAL}: {apdas}")


# %% Helpers
def year_color(year):
    if pd.isna(year):
        return "#cccccc"
    y = int(year)
    if y in ACRED_YEAR_COLORS:
        return ACRED_YEAR_COLORS[y]
    return FALLBACK_YEAR_COLORS[y % len(FALLBACK_YEAR_COLORS)]


def hover_text(sub):
    lines = []
    for _, row in sub.iterrows():
        dims = "<br>".join(f"{d}: {row[d]:.1f}" for d in valid_dims if pd.notna(row[d]))
        extra = ""
        if "dominant_archetype" in row and pd.notna(row.get("dominant_archetype")):
            extra = f"Arquetipo dominante: {row['dominant_archetype']} ({row['dominant_share']*100:.0f}%)<br>"
        lines.append(
            f"<b>{row['ies_label']}</b><br>"
            f"Años acreditación: {row[TARGET_COL]:.0f}<br>{extra}{dims}"
        )
    return lines


def year_legend_traces(sub, xkey, ykey, marker_symbol="circle"):
    """One legend-carrying scatter trace per año-acreditación category + UFT,
    with visible ies_label text next to each point (abreviación de IES)."""
    traces = []
    years_present = sorted(pd.to_numeric(sub[TARGET_COL], errors="coerce").dropna().astype(int).unique())
    for y in years_present:
        m = (pd.to_numeric(sub[TARGET_COL], errors="coerce") == y) & (~sub["_is_uft"])
        if not m.any():
            continue
        s = sub.loc[m]
        traces.append(go.Scatter(
            x=s[xkey], y=s[ykey], mode="markers+text",
            marker=dict(size=11, symbol=marker_symbol, color=year_color(y), line=dict(color="white", width=1)),
            text=s["ies_label"], textposition="top center", textfont=dict(size=7),
            name=f"{y} años", hovertext=hover_text(s), hoverinfo="text",
        ))
    if sub["_is_uft"].any():
        s = sub.loc[sub["_is_uft"]]
        traces.append(go.Scatter(
            x=s[xkey], y=s[ykey], mode="markers+text",
            marker=dict(size=13, symbol=marker_symbol, color=UFT_COLOR, line=dict(color="black", width=1)),
            text=s["ies_label"], textposition="top center", textfont=dict(size=8, color=UFT_COLOR),
            name="UFT", hovertext=hover_text(s), hoverinfo="text",
        ))
    return traces


def add_watermark(fig):
    """Same 'PRELIMINAR - NO REPRODUCIR' watermark the matplotlib scripts draw
    (Vector/01_vectors.py, Archetypes/02_archetypes.py), gated by config's
    watermark: flag. Uses paper-relative coords so it overlays cartesian,
    ternary, polar and 3D-scene figures alike."""
    if not WATERMARK:
        return
    fig.add_annotation(
        text="PRELIMINAR - NO REPRODUCIR", xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False, textangle=-30,
        font=dict(size=34, color="gray", family="Arial Black"), opacity=0.18,
    )


def base_layout(title):
    return dict(
        title=dict(text=title, x=0.02, xanchor="left"),
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(title="Años acreditación"),
        height=620,
    )


def build_axis(axis_cfg, df, valid_dims, apda_col=None, axes_by_apda=None):
    """Composite axis = NaN-aware weighted mean of dimension scores (0-100 scale).

    Thin wrapper around the shared `Utils/data_prep.py::build_axis` (single
    source of truth, also used by Vector/01_vectors.py and Sensitivity/) that
    unwraps its pd.Series return to a positional ndarray — this module's call
    sites (add_percentile_xy, compute_target_solve) rely on plain-ndarray,
    index-free alignment, not label-based alignment.

    `apda_col`/`axes_by_apda` passthrough to the shared implementation, for
    call sites that span more than one APDA at once (see _x_base/_y_base and
    compute_target_solve's perturbation loop below) — call sites already
    scoped to a single APDA's rows should instead resolve one flat axis_cfg
    via resolve_axis_cfg() and leave these unset.
    """
    scores, label = _shared_build_axis(axis_cfg, df, valid_dims, verbose=False,
                                        apda_col=apda_col, axes_by_apda=axes_by_apda)
    return (scores.to_numpy() if scores is not None else None), label


def fit_pca(df, valid_dims):
    # a dimension can be entirely NaN within this APDA's rows (e.g. a
    # dimension scoped to a single other APDA, like "Acreditación"
    # outside Salud y bienestar); SimpleImputer silently drops
    # such columns, so exclude them upfront and return the dims actually
    # used, keeping them aligned with pca.components_'s column count.
    valid_dims = [d for d in valid_dims if df[d].notna().any()]
    X = df[valid_dims].to_numpy(dtype=float)
    X_imp  = SimpleImputer(strategy=PCA_IMPUTE_STRATEGY).fit_transform(X)
    scaler = StandardScaler().fit(X_imp)
    X_std  = scaler.transform(X_imp)
    n_comp = min(PCA_N_COMPONENTS, X_std.shape[0] - 1, X_std.shape[1])
    pca = PCA(n_components=n_comp, random_state=PCA_RANDOM_STATE).fit(X_std)
    scores = pca.transform(X_std)
    return scores, pca, scaler, valid_dims


# ── Click-to-target (Mapa vectorial): minimal-change variable finder ────────
# raw variable -> normalize_variables_0_100 -> dimension score (weighted mean)
# -> build_axis (weighted mean) is affine end-to-end, so for each APDA there's
# a fixed Jacobian mapping "change in raw variable v" -> "change in (axis_x,
# axis_y)". Estimated numerically by perturbing UFT's raw value and re-running
# the real pipeline (compute_dimension_scores/build_axis) rather than
# re-deriving the chain rule by hand. _x_base/_y_base is only used as an
# internal, self-consistent reference for these perturbation deltas — the
# plotted map itself uses vec_sub (see compute_target_solve).
TARGET_EPS_FRAC = 0.01  # perturbation size, as a fraction of each variable's within-APDA std

with contextlib.redirect_stdout(io.StringIO()):
    _dims_base, _valid_dims_base = compute_dimension_scores(
        vec_features, vec_cfg["dimensions"], ID_COLS, target_col=TARGET_COL,
        direction=vec_cfg.get("direction", {}), apda_col="apda",
    )
_x_base, _ = build_axis(AXIS_X_CFG, _dims_base, _valid_dims_base,
                         apda_col="apda", axes_by_apda=AXIS_X_BY_APDA or None)
_y_base, _ = build_axis(AXIS_Y_CFG, _dims_base, _valid_dims_base,
                         apda_col="apda", axes_by_apda=AXIS_Y_BY_APDA or None)


def compute_target_solve(apda, vec_sub):
    """Minimum-total-std-dev raw-variable Jacobian for UFT's position on the
    'Mapa interactivo' composite axes (a click-enabled duplicate of 'Mapa
    vectorial', see build_interactive_map). The plotted x/y are rank-percentiles
    within the APDA (see add_percentile_xy) — not affine in the raw variables —
    so the solve happens in pre-rank composite-axis-score space; the clicked
    percentile is inverted back to that space client-side via sorted_x/sorted_y
    (linear interpolation over the APDA's population — see initTargetFinder).
    Returns None if UFT isn't present in this APDA (e.g. filtered out by
    ACRED_MIN_FILTER)."""
    if not vec_sub["_is_uft"].any():
        return None
    axis_x_cfg = resolve_axis_cfg(apda, "x")
    axis_y_cfg = resolve_axis_cfg(apda, "y")
    x_plot, _ = build_axis(axis_x_cfg, vec_sub, valid_dims)
    y_plot, _ = build_axis(axis_y_cfg, vec_sub, valid_dims)
    uft_pos = int(np.flatnonzero(vec_sub["_is_uft"].to_numpy())[0])
    if np.isnan(x_plot[uft_pos]) or np.isnan(y_plot[uft_pos]):
        return None
    x_rank = pd.Series(x_plot).rank(pct=True) * 100.0
    y_rank = pd.Series(y_plot).rank(pct=True) * 100.0
    uft_plot = [round(float(x_rank.iloc[uft_pos]), 4), round(float(y_rank.iloc[uft_pos]), 4)]

    feat_mask = (vec_features["apda"] == apda).to_numpy()
    feat_uft_idx = vec_features.index[feat_mask & vec_features["_is_uft"].to_numpy()]
    if len(feat_uft_idx) == 0:
        return None
    i = feat_uft_idx[0]

    apda_axis_dim_var = AXIS_DIM_VAR_BY_APDA[safe_id(apda)]
    cand_vars = sorted({v for vs in apda_axis_dim_var["x"]["dims"].values() for v in vs}
                        | {v for vs in apda_axis_dim_var["y"]["dims"].values() for v in vs})

    rows, kept_vars = [], []
    for var in cand_vars:
        raw_uft = vec_features.at[i, var]
        if pd.isna(raw_uft):
            continue
        std_v = vec_features.loc[feat_mask, var].std(ddof=0)
        if pd.isna(std_v) or std_v <= 1e-9:
            continue
        eps = TARGET_EPS_FRAC * std_v
        pert = vec_features.copy()
        pert.at[i, var] = raw_uft + eps
        with contextlib.redirect_stdout(io.StringIO()):
            dims_pert, valid_dims_pert = compute_dimension_scores(
                pert, vec_cfg["dimensions"], ID_COLS, target_col=TARGET_COL,
                direction=vec_cfg.get("direction", {}), apda_col="apda",
            )
        x_pert, _ = build_axis(AXIS_X_CFG, dims_pert, valid_dims_pert,
                                apda_col="apda", axes_by_apda=AXIS_X_BY_APDA or None)
        y_pert, _ = build_axis(AXIS_Y_CFG, dims_pert, valid_dims_pert,
                                apda_col="apda", axes_by_apda=AXIS_Y_BY_APDA or None)
        if np.isnan(x_pert[i]) or np.isnan(y_pert[i]):
            continue
        dx = (x_pert[i] - _x_base[i]) / eps * std_v
        dy = (y_pert[i] - _y_base[i]) / eps * std_v
        rows.append((dx, dy))
        kept_vars.append(var)

    if not rows:
        return None
    J = np.array(rows)  # n_vars x 2
    try:
        M = J @ np.linalg.pinv(J.T @ J)  # n_vars x 2, minimum-norm solve matrix
    except np.linalg.LinAlgError:
        return None

    sorted_x = sorted(float(v) for v in x_plot if not np.isnan(v))
    sorted_y = sorted(float(v) for v in y_plot if not np.isnan(v))

    return {
        "vars": kept_vars,
        "M": [[round(float(a), 6), round(float(b), 6)] for a, b in M],
        "orig_axis": [round(float(x_plot[uft_pos]), 6), round(float(y_plot[uft_pos]), 6)],
        "uft_plot": uft_plot,
        "sorted_x": [round(v, 6) for v in sorted_x],
        "sorted_y": [round(v, 6) for v in sorted_y],
        "raw_current": {v: round(float(vec_features.at[i, v]), 4) for v in kept_vars},
        "raw_std": {v: round(float(vec_features.loc[feat_mask, v].std(ddof=0)), 6) for v in kept_vars},
        "direction": {v: int(vec_cfg.get("direction", {}).get(v, 1)) for v in kept_vars},
    }


def add_loading_arrows(fig, pca, inst_scores, valid_dims):
    load_max  = max(np.abs(pca.components_[0]).max(), np.abs(pca.components_[1]).max(), 1e-9)
    score_max = max(np.abs(inst_scores[:, 0]).max(), np.abs(inst_scores[:, 1]).max(), 1e-9)
    arrow_scale = ARROW_SCALE_FACTOR * score_max / load_max
    for j, dim_name in enumerate(valid_dims):
        ax_end = pca.components_[0, j] * arrow_scale
        ay_end = pca.components_[1, j] * arrow_scale
        fig.add_annotation(x=ax_end, y=ay_end, ax=0, ay=0, xref="x", yref="y",
                            axref="x", ayref="y", showarrow=True, arrowhead=2,
                            arrowsize=1, arrowwidth=1.3, arrowcolor="crimson", opacity=0.55)
        fig.add_annotation(x=ax_end * 1.15, y=ay_end * 1.15, text=dim_name,
                            showarrow=False, font=dict(size=10, color="crimson"),
                            bgcolor="white", opacity=0.85)


# %% Chart builders
def add_percentile_xy(sub, apda=None):
    """Rank each institution's composite axis score into a 0-100 percentile
    within its APDA — the coordinate system both build_vector_map and
    build_interactive_map plot in (see compute_target_solve for why this
    matters: it's not affine in the raw variables)."""
    axis_x_cfg = resolve_axis_cfg(apda, "x") if apda is not None else AXIS_X_CFG
    axis_y_cfg = resolve_axis_cfg(apda, "y") if apda is not None else AXIS_Y_CFG
    x_vals, x_label = build_axis(axis_x_cfg, sub, valid_dims)
    y_vals, y_label = build_axis(axis_y_cfg, sub, valid_dims)
    sub = sub.reset_index(drop=True).copy()
    sub["_x"] = pd.Series(x_vals).rank(pct=True) * 100.0
    sub["_y"] = pd.Series(y_vals).rank(pct=True) * 100.0
    return sub, x_label, y_label


def _build_percentile_map(apda, sub, title):
    sub, x_label, y_label = add_percentile_xy(sub, apda=apda)
    fig = go.Figure()
    for tr in year_legend_traces(sub, "_x", "_y"):
        fig.add_trace(tr)
    fig.add_hline(y=50, line=dict(color="#cccccc", dash="dash", width=0.8))
    fig.add_vline(x=50, line=dict(color="#cccccc", dash="dash", width=0.8))
    fig.update_layout(**base_layout(title))
    fig.update_xaxes(title=AXIS_LABELS.get("x", x_label) + f"  (percentil {GROUP_WITHIN})", range=list(AXIS_LIMITS))
    fig.update_yaxes(title=AXIS_LABELS.get("y", y_label) + f"  (percentil {GROUP_WITHIN})", range=list(AXIS_LIMITS))
    add_watermark(fig)
    return fig


def build_vector_map(apda, sub):
    return _build_percentile_map(apda, sub, f"Mapa vectorial — {apda}  (N={len(sub)})")


def build_interactive_map(apda, sub):
    """Same plot as build_vector_map (same data, same percentile axes), kept
    as an independent figure/div so the click-to-target feature (red dot +
    arrow, wired up in initTargetFinder) doesn't touch the original 'Mapa
    vectorial' tab."""
    return _build_percentile_map(apda, sub, f"Mapa interactivo — {apda}  (N={len(sub)})")


def build_pca_scatter(apda, sub, valid_dims, profiles=None):
    scores, pca, scaler, valid_dims = fit_pca(sub, valid_dims)
    var_exp = pca.explained_variance_ratio_
    sub = sub.reset_index(drop=True).copy()
    sub["_pc1"] = scores[:, 0]
    sub["_pc2"] = scores[:, 1]

    fig = go.Figure()
    for tr in year_legend_traces(sub, "_pc1", "_pc2"):
        fig.add_trace(tr)

    title = f"PCA — {apda}  (N={len(sub)})"
    if profiles is not None:
        k = len(profiles)
        arche_std = scaler.transform(profiles[valid_dims].to_numpy(dtype=float))
        arche_scores = pca.transform(arche_std)

        if k >= 3:
            try:
                hull = ConvexHull(arche_scores[:, :2])
                order = list(hull.vertices) + [hull.vertices[0]]
                fig.add_trace(go.Scatter(
                    x=arche_scores[order, 0], y=arche_scores[order, 1],
                    mode="lines", line=dict(color="black", dash="dash", width=1.3),
                    showlegend=False, hoverinfo="skip",
                ))
            except QhullError:
                pass
        elif k == 2:
            fig.add_trace(go.Scatter(
                x=arche_scores[:, 0], y=arche_scores[:, 1],
                mode="lines", line=dict(color="black", dash="dash", width=1.3),
                showlegend=False, hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=arche_scores[:, 0], y=arche_scores[:, 1], mode="markers+text",
            marker=dict(symbol="star", size=20, color="gold", line=dict(color="black", width=1.3)),
            text=profiles["archetype"], textposition="top center",
            textfont=dict(size=13, color="black"),
            name="Arquetipos (AA)", hovertext=profiles["archetype"], hoverinfo="text",
        ))
        title = f"PCA + AA — {apda}  (k={k}, N={len(sub)})"

    add_loading_arrows(fig, pca, scores, valid_dims)

    fig.add_hline(y=0, line=dict(color="#cccccc", dash="dash", width=0.8))
    fig.add_vline(x=0, line=dict(color="#cccccc", dash="dash", width=0.8))
    fig.update_layout(**base_layout(title))
    fig.update_xaxes(title=f"PC1  ({var_exp[0]*100:.1f} % varianza explicada)")
    fig.update_yaxes(title=f"PC2  ({var_exp[1]*100:.1f} % varianza explicada)" if len(var_exp) > 1 else "PC2")
    add_watermark(fig)
    return fig


def build_ternary(apda, sub):
    sub = sub.reset_index(drop=True)
    fig = go.Figure()
    years_present = sorted(pd.to_numeric(sub[TARGET_COL], errors="coerce").dropna().astype(int).unique())
    for y in years_present:
        m = (pd.to_numeric(sub[TARGET_COL], errors="coerce") == y) & (~sub["_is_uft"])
        if not m.any():
            continue
        s = sub.loc[m]
        fig.add_trace(go.Scatterternary(
            a=s["alpha_1"], b=s["alpha_2"], c=s["alpha_3"], mode="markers+text",
            marker=dict(size=11, color=year_color(y), line=dict(color="white", width=1)),
            text=s["ies_label"], textposition="top center", textfont=dict(size=7),
            name=f"{y} años", hovertext=hover_text(s), hoverinfo="text",
        ))
    if sub["_is_uft"].any():
        s = sub.loc[sub["_is_uft"]]
        fig.add_trace(go.Scatterternary(
            a=s["alpha_1"], b=s["alpha_2"], c=s["alpha_3"], mode="markers+text",
            marker=dict(size=13, color=UFT_COLOR, line=dict(color="black", width=1)),
            text=s["ies_label"], textposition="top center", textfont=dict(size=8, color=UFT_COLOR),
            name="UFT", hovertext=hover_text(s), hoverinfo="text",
        ))
    fig.update_layout(
        **base_layout(f"Simplex de membresía (ternario) — {apda}"),
        ternary=dict(aaxis=dict(title="A1"), baxis=dict(title="A2"), caxis=dict(title="A3")),
    )
    add_watermark(fig)
    return fig


def build_tetrahedron(apda, sub):
    sub = sub.reset_index(drop=True)
    alphas = sub[["alpha_1", "alpha_2", "alpha_3", "alpha_4"]].to_numpy(dtype=float)
    points = alphas @ TETRA_VERTS
    sub = sub.copy()
    sub["_x3"], sub["_y3"], sub["_z3"] = points[:, 0], points[:, 1], points[:, 2]

    fig = go.Figure()
    for i, j in TETRA_EDGES:
        fig.add_trace(go.Scatter3d(
            x=[TETRA_VERTS[i, 0], TETRA_VERTS[j, 0]],
            y=[TETRA_VERTS[i, 1], TETRA_VERTS[j, 1]],
            z=[TETRA_VERTS[i, 2], TETRA_VERTS[j, 2]],
            mode="lines", line=dict(color="#999999", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    years_present = sorted(pd.to_numeric(sub[TARGET_COL], errors="coerce").dropna().astype(int).unique())
    for y in years_present:
        m = (pd.to_numeric(sub[TARGET_COL], errors="coerce") == y) & (~sub["_is_uft"])
        if not m.any():
            continue
        s = sub.loc[m]
        fig.add_trace(go.Scatter3d(
            x=s["_x3"], y=s["_y3"], z=s["_z3"], mode="markers+text",
            marker=dict(size=5, color=year_color(y)),
            text=s["ies_label"], textfont=dict(size=7),
            name=f"{y} años", hovertext=hover_text(s), hoverinfo="text",
        ))
    if sub["_is_uft"].any():
        s = sub.loc[sub["_is_uft"]]
        fig.add_trace(go.Scatter3d(
            x=s["_x3"], y=s["_y3"], z=s["_z3"], mode="markers+text",
            marker=dict(size=6, color=UFT_COLOR),
            text=s["ies_label"], textfont=dict(size=8, color=UFT_COLOR),
            name="UFT", hovertext=hover_text(s), hoverinfo="text",
        ))
    for (vx, vy, vz), label in zip(TETRA_VERTS * 1.2, ["A1", "A2", "A3", "A4"]):
        fig.add_trace(go.Scatter3d(
            x=[vx], y=[vy], z=[vz], mode="text", text=[label],
            textfont=dict(size=14, color="crimson"), showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        **base_layout(f"Simplex de arquetipos (tetraedro) — {apda}"),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
    )
    add_watermark(fig)
    return fig


def build_radar(apda, profiles, valid_dims):
    # a dimension can be absent from this APDA's archetype profiles (e.g. a
    # dimension scoped to a single other APDA, like "Acreditación"
    # outside Salud y bienestar, gets dropped upstream in
    # Archetypes/02_archetypes.py when it's entirely NaN there).
    valid_dims = [d for d in valid_dims if d in profiles.columns]
    fig = go.Figure()
    theta = valid_dims + [valid_dims[0]]
    for i, (_, row) in enumerate(profiles.iterrows()):
        vals = [row[d] for d in valid_dims]
        vals += vals[:1]
        color = ARCHETYPE_COLORS[i % len(ARCHETYPE_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta, name=row["archetype"], mode="lines+markers",
            line=dict(color=color), fill="toself", opacity=0.75,
        ))
    fig.update_layout(**base_layout(f"Perfil de arquetipos — {apda}"),
                       polar=dict(radialaxis=dict(range=[0, 100])))
    add_watermark(fig)
    return fig


def build_stacked_bar(apda, sub, k):
    sub = sub.sort_values(["dominant_archetype", "dominant_share"], ascending=[True, False]).reset_index(drop=True)
    fig = go.Figure()
    for i in range(k):
        col = f"alpha_{i+1}"
        fig.add_trace(go.Bar(
            x=sub["ies_label"], y=sub[col], name=f"A{i+1}",
            marker=dict(color=ARCHETYPE_COLORS[i % len(ARCHETYPE_COLORS)]),
            hovertext=[f"{lbl}: {v*100:.0f}%" for lbl, v in zip(sub['ies_label'], sub[col])],
            hoverinfo="text",
        ))
    fig.update_layout(**base_layout(f"Membresía a arquetipos — {apda}  (k={k})"),
                       barmode="stack", legend_title="Arquetipo")
    fig.update_yaxes(title="Peso de membresía (alpha)", range=[0, 1.02])
    ticktext = [
        f'<span style="color:{UFT_COLOR}"><b>{lbl}</b></span>' if is_uft else lbl
        for lbl, is_uft in zip(sub["ies_label"], sub["_is_uft"])
    ]
    fig.update_xaxes(tickangle=90, tickfont=dict(size=9),
                      tickmode="array", tickvals=sub["ies_label"], ticktext=ticktext)
    if sub["_is_uft"].any():
        uft_label = sub.loc[sub["_is_uft"], "ies_label"].iloc[0]
        fig.add_annotation(x=uft_label, y=1.05, text="UFT", showarrow=True,
                            arrowhead=2, ax=0, ay=-30,
                            font=dict(color=UFT_COLOR, size=12, family="Arial Black"),
                            arrowcolor=UFT_COLOR)
    add_watermark(fig)
    return fig


def build_variable_bar(apda, var_sub, all_vars, default_var):
    """Bar chart of one variable across institutions, sorted descending
    (by normalized score) with UFT highlighted. Only a single go.Bar trace
    is ever rendered — the page's JS switches variables and normalized/raw
    scale client-side (Plotly.restyle/relayout) using the returned var_data
    dict, precomputed here for every variable so no server round-trip is
    needed for the axis -> dimension -> variable selector or the scale toggle."""
    var_data = {}
    for var in all_vars:
        col = f"{var}__norm"
        if col not in var_sub.columns or var not in var_sub.columns:
            continue
        s = var_sub[["ies_label", "_is_uft", var, col]].dropna(subset=[col])
        if s.empty:
            continue
        s = s.sort_values(col, ascending=False)
        colors = [UFT_COLOR if is_uft else BAR_COLOR for is_uft in s["_is_uft"]]
        ticktext = [
            f'<span style="color:{UFT_COLOR}"><b>{lbl}</b></span>' if is_uft else lbl
            for lbl, is_uft in zip(s["ies_label"], s["_is_uft"])
        ]
        y_norm = s[col].round(1).tolist()
        y_raw = s[var].round(2).tolist()
        var_data[var] = {
            "x": s["ies_label"].tolist(),
            "y_norm": y_norm,
            "y_raw": y_raw,
            "colors": colors,
            "ticktext": ticktext,
            "title": f"{display_name(var, var_glossary)} — {apda}",
            "hovertext_norm": [f"{lbl}: {v:.1f}" for lbl, v in zip(s["ies_label"], y_norm)],
            "hovertext_raw": [f"{lbl}: {v:.2f}" for lbl, v in zip(s["ies_label"], y_raw)],
        }

    default_var = default_var if default_var in var_data else next(iter(var_data), None)
    d0 = var_data.get(default_var, {"x": [], "y_norm": [], "colors": [], "ticktext": [],
                                     "title": apda, "hovertext_norm": []})

    fig = go.Figure(go.Bar(
        x=d0["x"], y=d0["y_norm"], marker=dict(color=d0["colors"]),
        hovertext=d0["hovertext_norm"], hoverinfo="text",
    ))
    fig.update_layout(**base_layout(d0["title"]))
    fig.update_yaxes(title="Score normalizado (0-100)", range=[0, 100])
    fig.update_xaxes(tickangle=90, tickfont=dict(size=9),
                      tickmode="array", tickvals=d0["x"], ticktext=d0["ticktext"])
    add_watermark(fig)
    return fig, var_data


# %% Build per-APDA figures and HTML fragments
def fig_to_div(fig, div_id=None):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                        config={"responsive": True}, div_id=div_id or str(uuid.uuid4()))


def build_dim_score_payload(sub, valid_dims):
    """Per-institution dimension scores for this APDA, grouped and ordered
    exactly like year_legend_traces (one group per año-acreditación category,
    then UFT) — so group index i here lines up with trace index i in both the
    'Mapa vectorial' and 'Mapa interactivo' figures. Used client-side by
    updateAxisWeights() to recompute axis positions when the weight table is
    edited, without a server round-trip: dimension scores themselves don't
    depend on axis weights (see Utils/data_prep.py::build_axis), only the
    weighted mean and the percentile rank that follow do."""
    years_present = sorted(pd.to_numeric(sub[TARGET_COL], errors="coerce").dropna().astype(int).unique())
    rows, group_sizes = [], []
    for y in years_present:
        m = (pd.to_numeric(sub[TARGET_COL], errors="coerce") == y) & (~sub["_is_uft"])
        if not m.any():
            continue
        s = sub.loc[m]
        group_sizes.append(len(s))
        rows.extend({d: (None if pd.isna(row[d]) else float(row[d])) for d in valid_dims}
                     for _, row in s.iterrows())
    if sub["_is_uft"].any():
        s = sub.loc[sub["_is_uft"]]
        group_sizes.append(len(s))
        rows.extend({d: (None if pd.isna(row[d]) else float(row[d])) for d in valid_dims}
                     for _, row in s.iterrows())
    return {"rows": rows, "group_sizes": group_sizes}


def build_axis_weight_bar(axis_cfg, title):
    """Horizontal bar of each dimension's normalised weight (%) within an axis."""
    dims = [d for d in axis_cfg if d in valid_dims]
    weights = np.array([axis_cfg[d] for d in dims], dtype=float)
    pct = weights / weights.sum() * 100.0
    order = np.argsort(pct)
    dims = [dims[i] for i in order]
    pct = pct[order]
    fig = go.Figure(go.Bar(
        x=pct, y=dims, orientation="h",
        marker=dict(color="#3881BC"),
        text=[f"{p:.1f} %" for p in pct], textposition="outside",
        hoverinfo="text", hovertext=[f"{d}: {p:.1f} %" for d, p in zip(dims, pct)],
    ))
    fig.update_layout(**base_layout(title))
    fig.update_xaxes(title="Peso en el eje (%)", range=[0, max(pct) * 1.2])
    fig.update_yaxes(title="")
    fig.update_layout(height=max(280, 55 * len(dims)))
    add_watermark(fig)
    return fig


def build_dimension_table():
    """HTML table: each dimension and the raw variables that compose it,
    flagging variables whose orientation is inverted (direction: -1 in
    Vector/config.yaml, e.g. costo, año de inicio — 'menor es mejor')."""
    direction_cfg = vec_cfg.get("direction", {})
    rows = []
    for dim, spec in vec_cfg["dimensions"].items():
        if dim not in valid_dims:
            continue
        variables = spec.get("variables", {})
        items = []
        for var in variables:
            inv = direction_cfg.get(var, 1) == -1
            flag = ' <span class="inv-flag" title="Orientación invertida: menor valor = mejor">&#8645; invertida</span>' if inv else ""
            items.append(f"<li>{display_name(var, var_glossary)}{flag}{info_icon_html(var)}</li>")
        rows.append(f"<tr><td class='dim-name'>{dim}</td><td><ul class='var-list'>{''.join(items)}</ul></td></tr>")
    return f"""
    <table class="dim-table">
      <thead><tr><th>Dimensión</th><th>Variables que la componen</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def build_pca_similarity_heatmap(loadings_df, pc):
    """Cross-APDA |cosine similarity| between loading vectors for one
    component — mirrors Vector/01_vectors.py's '# %% PCA composition
    comparison' cell. 1.0 = same dimensions drive the component in both
    APDAs (same direction, up to an arbitrary sign flip); 0.0 = unrelated
    compositions."""
    pivot = loadings_df[loadings_df["PC"] == pc].pivot(
        index="dimension", columns="apda", values="loading").reindex(valid_dims)
    mat = np.nan_to_num(pivot.to_numpy(), nan=0.0)
    norms = np.linalg.norm(mat, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    sim = np.abs(unit.T @ unit)
    cols = list(pivot.columns)

    fig = go.Figure(go.Heatmap(
        z=sim, x=cols, y=cols, zmin=0, zmax=1, colorscale="Viridis",
        text=[[f"{v:.2f}" for v in row] for row in sim], texttemplate="%{text}",
        hoverinfo="text",
        hovertext=[[f"{a} vs {b}: {v:.2f}" for b, v in zip(cols, row)] for a, row in zip(cols, sim)],
        colorbar=dict(title="|cos sim|"),
    ))
    fig.update_layout(**base_layout(f"Similitud entre {GROUP_PLURAL} — composición de {pc}"))
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=max(360, 70 * len(cols) + 120))
    add_watermark(fig)
    return fig


tab_sections = []
tab_buttons = []
bi_sections = {}  # apda -> standalone section HTML, for Dashboard/BI testing/{apda}.html
site_sections = {}  # apda -> standalone section HTML (4 map views), for Dashboard/site/{apda|cine}/

# ── Welcome tab: dimension composition + axis weight breakdown ────────────────
axis_x_div = fig_to_div(build_axis_weight_bar(AXIS_X_CFG, f"Eje X — {AXIS_LABELS.get('x', 'X')}: peso por dimensión"))
axis_y_div = fig_to_div(build_axis_weight_bar(AXIS_Y_CFG, f"Eje Y — {AXIS_LABELS.get('y', 'Y')}: peso por dimensión"))

loadings_records = []
for apda in apdas:
    sub = vec_dim_scores[vec_dim_scores["apda"] == apda]
    if len(sub) <= 2:
        continue
    _, pca, _, local_dims = fit_pca(sub, valid_dims)
    for i, pc in enumerate(("PC1", "PC2")):
        if i >= pca.components_.shape[0]:
            continue
        for j, dim in enumerate(local_dims):
            loadings_records.append({"apda": apda, "PC": pc, "dimension": dim,
                                      "loading": pca.components_[i, j]})

loadings_df = pd.DataFrame(loadings_records)
similarity_divs = [fig_to_div(build_pca_similarity_heatmap(loadings_df, pc)) for pc in ("PC1", "PC2")] \
    if not loadings_df.empty else []

intro_note_block = f"""
      <div class="chart-block">
        <h3>{GROUP_LABEL_CAP} como unidad de análisis</h3>
        <p class="note">{BRANCH['intro_note']}</p>
      </div>""" if BRANCH["intro_note"] else ""

welcome_section = f"""
    <div class="tab-content active" id="tab-inicio">
      {intro_note_block}
      <div class="chart-block">
        <h3>Composición de dimensiones</h3>
        <p class="note">Cada dimensión resume una o más variables originales del dataset
          (normalizadas por z-score dentro de {GROUP_EACH}). Las variables marcadas como
          "invertida" tienen orientación natural opuesta a "mayor = mejor" (p. ej. costo,
          año de inicio del programa) y se invierten antes de normalizar, de modo que en
          toda dimensión un valor más alto siempre es mejor.</p>
        {build_dimension_table()}
      </div>
      <div class="chart-block">
        <h3>Composición de los ejes (Mapa vectorial)</h3>
        <p class="note">El "Mapa vectorial" ubica cada institución según dos ejes compuestos:
          {AXIS_LABELS.get('x', 'eje X')} (eje X) y {AXIS_LABELS.get('y', 'eje Y')} (eje Y). Cada eje es un promedio
          ponderado (NaN-aware) de las dimensiones de arriba; los pesos —definidos en
          <code>Vector/config.yaml</code>— se muestran a continuación como porcentaje
          normalizado del eje.</p>
        {axis_x_div}
        {axis_y_div}
      </div>
      <div class="chart-block">
        <h3>Similitud entre {GROUP_PLURAL} — composición del PCA</h3>
        <p class="note">{GROUP_EACH_CAP} ajusta su propio PCA por separado, así que PC1/PC2 pueden
          estar compuestos por dimensiones distintas en cada uno. Estos mapas comparan,
          componente por componente, qué tan parecida es esa composición entre {GROUP_PLURAL} usando
          |similitud coseno| entre los vectores de loadings: 1 = las mismas dimensiones
          dominan el componente en las dos (misma dirección, salvo signo — el signo del
          componente en sí es arbitrario); 0 = composiciones sin relación.</p>
        {''.join(similarity_divs)}
      </div>
    </div>"""
welcome_button = '<button class="tab-btn active" onclick="showTab(\'tab-inicio\', this)">Inicio</button>'

# ── Decisiones metodológicas tab: prose lives in build_dashboard_texto.md ─────
DECISIONES_TEXTS = load_text_blocks(BRANCH["texto"])
decisiones_section = f"""
    <div class="tab-content" id="tab-decisiones">
      <div class="chart-block">
        {md_to_html(DECISIONES_TEXTS['decisiones_intro'])}
      </div>
    </div>"""
decisiones_button = '<button class="tab-btn" onclick="showTab(\'tab-decisiones\', this)">Decisiones metodológicas</button>'

for apda in apdas:
    safe = safe_id(apda)
    safe_js = json.dumps(safe)
    tab_id = f"tab-{safe}"
    if apda in DASHBOARD_K_OVERRIDE:
        profiles = k4_profiles[apda]
    else:
        profile_path = ARCH_PROFILE_TMPL.format(safe=safe)
        if not os.path.exists(profile_path):
            print(f"[{apda}] Missing {profile_path}, skipping.")
            continue
        profiles = pd.read_csv(profile_path, encoding="utf-8-sig")
    k = len(profiles)

    arch_sub = merged[merged["apda"] == apda].reset_index(drop=True)
    vec_sub  = vec_dim_scores[vec_dim_scores["apda"] == apda].reset_index(drop=True)
    print(f"[{apda}] N(archetypes)={len(arch_sub)}, N(vector)={len(vec_sub)}, k={k}")

    # ── Per-APDA editable weight table (settings button, lives in this APDA's
    # own tab — not attached to any one chart). Edits recompute the 'Mapa
    # vectorial'/'Mapa interactivo' positions client-side (see updateAxisWeights).
    axis_x_cfg_apda = resolve_axis_cfg(apda, "x")
    axis_y_cfg_apda = resolve_axis_cfg(apda, "y")
    weights_x_dims = sorted(d for d in axis_x_cfg_apda if d in valid_dims)
    weights_y_dims = sorted(d for d in axis_y_cfg_apda if d in valid_dims)
    weights_rows_x = "".join(
        f'<tr><td>{d}</td><td><input type="number" step="0.1" min="0" value="{axis_x_cfg_apda[d]}" '
        f'data-dim="{d}" oninput="updateAxisWeights(\'{safe}\',\'x\')"></td></tr>'
        for d in weights_x_dims
    )
    weights_rows_y = "".join(
        f'<tr><td>{d}</td><td><input type="number" step="0.1" min="0" value="{axis_y_cfg_apda[d]}" '
        f'data-dim="{d}" oninput="updateAxisWeights(\'{safe}\',\'y\')"></td></tr>'
        for d in weights_y_dims
    )
    dim_score_payload = build_dim_score_payload(vec_sub, valid_dims)
    weights_panel_html = f"""
        <div class="weights-panel" id="weights-panel-{safe}" style="display:none;">
          <div class="weights-axis">
            <h4>Eje X — {AXIS_LABELS.get('x', 'X')}</h4>
            <table class="weights-table" id="weights-table-x-{safe}">
              <thead><tr><th>Dimensión</th><th>Peso (%)</th></tr></thead>
              <tbody>{weights_rows_x}</tbody>
            </table>
            <div class="weights-total" id="weights-total-x-{safe}"></div>
          </div>
          <div class="weights-axis">
            <h4>Eje Y — {AXIS_LABELS.get('y', 'Y')}</h4>
            <table class="weights-table" id="weights-table-y-{safe}">
              <thead><tr><th>Dimensión</th><th>Peso (%)</th></tr></thead>
              <tbody>{weights_rows_y}</tbody>
            </table>
            <div class="weights-total" id="weights-total-y-{safe}"></div>
          </div>
          <button class="reset-btn" onclick="resetAxisWeights('{safe}')">Restablecer valores por defecto</button>
          <p class="note" id="target-stale-{safe}" style="display:none;">Ajustaste los pesos: el
            buscador de objetivo del "Mapa interactivo" queda desactivado hasta que los
            restablezcas.</p>
        </div>"""
    weights_score_script = f"""
      <script>
        DIM_SCORES_BY_APDA[{safe_js}] = {json.dumps(dim_score_payload)};
        AXIS_WEIGHTS[{safe_js}] = {json.dumps({"x": axis_x_cfg_apda, "y": axis_y_cfg_apda}, ensure_ascii=False)};
        window.addEventListener('DOMContentLoaded', () => initWeightsTotals({safe_js}));
      </script>"""

    # ── Map: 4-way toggle (Mapa vectorial / Mapa interactivo / PCA / PCA + AA) ─
    vector_fig      = build_vector_map(apda, vec_sub)
    interactive_fig = build_interactive_map(apda, vec_sub)
    pca_fig         = build_pca_scatter(apda, vec_sub, valid_dims, profiles=None)
    pcaaa_fig       = build_pca_scatter(apda, arch_sub, valid_dims, profiles=profiles)

    vector_div_id      = f"vecmap-{safe}"
    interactive_div_id = f"mapinteractivo-{safe}"
    vector_div      = fig_to_div(vector_fig, div_id=vector_div_id)
    interactive_div = fig_to_div(interactive_fig, div_id=interactive_div_id)
    pca_div         = fig_to_div(pca_fig)
    pcaaa_div       = fig_to_div(pcaaa_fig)

    target_solve = compute_target_solve(apda, vec_sub)
    target_script = ""
    target_result_html = ""
    if target_solve is not None:
        target_result_html = f"""
        <p class="note">Click en un punto del mapa para ver la ruta de cambio total mínimo
          (en desviaciones estándar respecto a su dispersión dentro de {GROUP_THIS}) que llevaría
          a UFT ahí. Se muestran las variables que más deben ajustarse dentro de esa ruta —
          las de cambio despreciable quedan fuera. Aproximación lineal local: supone fijas las
          estadísticas de normalización del resto de las instituciones.</p>
        <div id="target-result-{safe}" class="target-result"></div>"""
        target_script = f"""
      <script>
        TARGET_SOLVE[{safe_js}] = {json.dumps(target_solve)};
        window.addEventListener('DOMContentLoaded', () => initTargetFinder({safe_js}));
      </script>"""

    map_group_id = f"mapgroup-{safe}"
    all_map_views = [
        ("Mapa vectorial",   "vector",      vector_div),
        ("Mapa interactivo", "interactivo", interactive_div + target_result_html),
        ("PCA",              "pca",         pca_div),
        ("PCA + AA",         "pcaaa",       pcaaa_div),
    ]

    def _map_block(views):
        """Bloque 'Mapa de instituciones' con un subtab por vista; el primero
        queda activo. `views` es una lista de (label, slug, contenido_html)."""
        buttons = "".join(
            f'<button class="subtab-btn{" active" if i == 0 else ""}" '
            f'onclick="showSubTab(\'{map_group_id}\',\'{map_group_id}-{slug}\', this)">{label}</button>'
            for i, (label, slug, _) in enumerate(views)
        )
        contents = "".join(
            f'<div class="subtab-content{" active" if i == 0 else ""}" id="{map_group_id}-{slug}">{content}</div>'
            for i, (label, slug, content) in enumerate(views)
        )
        return f"""
      <div class="chart-block">
        <div class="weights-header">
          <h3>Mapa de instituciones</h3>
          <button class="settings-btn" onclick="toggleWeightsPanel('{safe}')">&#9881; Ajustar pesos</button>
        </div>
        <div class="map-with-weights">
          {weights_panel_html}
          <div class="map-main">
            <div class="subtab-buttons" id="{map_group_id}">
              {buttons}
            </div>
            {contents}
          </div>
        </div>
      </div>{target_script}"""

    map_block    = _map_block(all_map_views)      # index.html: las 4 vistas
    map_block_bi = _map_block(all_map_views[:2])  # BI testing: sin PCA / PCA + AA

    # ── Variable individual: axis -> dimension -> variable cascading selector ─
    var_sub = vec_features[vec_features["apda"] == apda].merge(vec_sub[ID_COLS], on=ID_COLS, how="inner")
    var_bar_fig, var_data = build_variable_bar(apda, var_sub, bar_all_vars, bar_default_var)
    var_bar_div_id = f"varbar-{safe}"
    var_bar_div = fig_to_div(var_bar_fig, div_id=var_bar_div_id)
    var_selector_block = f"""
      <div class="chart-block">
        <h3>Variable individual por institución</h3>
        <p class="note">Score normalizado 0-100 (mismo método que las dimensiones: z-score
          {GROUP_WITHIN}, reescalado globalmente, orientación corregida) o valor original,
          según el toggle de la derecha. Las barras se ordenan de mayor a menor según el score
          normalizado de la variable seleccionada; UFT se destaca en color.</p>
        <div class="var-selector">
          <select id="sel-axis-{safe}" onchange="onAxisChange('{safe}')"></select>
          <select id="sel-dim-{safe}" onchange="onDimChange('{safe}')"></select>
          <select id="sel-var-{safe}" onchange="onVarChange('{safe}')"></select>
          <span class="info-icon">&#128712;<span class="tooltip-box" id="varinfo-{safe}">Seleccione una variable.</span></span>
          <div class="scale-toggle" id="scaletoggle-{safe}" style="margin-left:auto;">
            <button class="subtab-btn active" onclick="onScaleChange('{safe}','norm', this)">Normalizado</button>
            <button class="subtab-btn" onclick="onScaleChange('{safe}','raw', this)">Original</button>
          </div>
        </div>
        {var_bar_div}
      </div>
      <script>
        VAR_DATA[{safe_js}] = {json.dumps(var_data)};
        window.addEventListener('DOMContentLoaded', () => initVarSelector({safe_js}));
      </script>"""

    # ── Simplex: ternary for k=3, tetrahedron for k=4, note otherwise ─────────
    if k == 3:
        simplex_html = f"""
        <div class="chart-block">
          <h3>Simplex de membresía (ternario)</h3>
          {fig_to_div(build_ternary(apda, arch_sub))}
        </div>"""
    elif k == 4:
        simplex_html = f"""
        <div class="chart-block">
          <h3>Simplex de membresía (tetraedro 3D)</h3>
          {fig_to_div(build_tetrahedron(apda, arch_sub))}
        </div>"""
    else:
        simplex_html = f"""
        <div class="chart-block">
          <p class="note">Simplex geométrico exacto no disponible para k={k} (solo k=3 o k=4).</p>
        </div>"""

    radar_div = fig_to_div(build_radar(apda, profiles, valid_dims))
    bar_div   = fig_to_div(build_stacked_bar(apda, arch_sub, k))

    def _section(cls, map_html):
        """Cuerpo de esta APDA (tabla de pesos, mapa, selector de variable,
        simplex, radar, membresía); cls fija si la seccion arranca visible."""
        return f"""
    <div class="{cls}" id="{tab_id}">
      {weights_score_script}
      {map_html}
      {var_selector_block}
      {simplex_html}
      <div class="chart-block">
        <h3>Perfil de arquetipos (radar)</h3>
        {radar_div}
      </div>
      <div class="chart-block">
        <h3>Membresía por institución</h3>
        {bar_div}
      </div>
    </div>"""

    tab_sections.append(_section("tab-content", map_block))
    tab_buttons.append(f'<button class="tab-btn" onclick="showTab(\'{tab_id}\', this)">{apda}</button>')
    bi_sections[apda] = _section("tab-content active", map_block_bi)
    site_sections[apda] = _section("tab-content active", map_block)

# %% Assemble page
# Con parent_groups la barra queda de dos niveles: arriba el grupo padre
# (junto a Inicio/Decisiones), abajo los grupos hijos del padre elegido, que
# JS reconstruye en cada clic. Sin parent_groups queda la barra unica de
# siempre. En los dos casos las secciones ya vienen todas renderizadas y solo
# se alterna cual esta visible.
PARENT_GROUPS_JS = {
    parent: [[g, safe_id(g)] for g in groups if g in _present]
    for parent, groups in (PARENT_GROUPS or {}).items()
}
PARENT_GROUPS_JS = {p: gs for p, gs in PARENT_GROUPS_JS.items() if gs}

if PARENT_GROUPS_JS:
    all_buttons = [welcome_button, decisiones_button,
                   f'<span class="row-label">{PARENT_LABEL}:</span>'] + [
        f"<button class=\"tab-btn\" onclick=\"showParent('{p}', this)\">{p}</button>"
        for p in PARENT_GROUPS_JS
    ]
    group_row_html = (f'<div class="tabs group-row" id="group-row" style="display:none;">'
                      f'<span class="row-label">{GROUP_LABEL}:</span></div>')
else:
    all_buttons = [welcome_button, decisiones_button] + tab_buttons
    group_row_html = ""

all_sections = [welcome_section, decisiones_section] + tab_sections
parent_groups_json = json.dumps(PARENT_GROUPS_JS, ensure_ascii=False)
page_title   = BRANCH["title"].format(date=date.today().strftime("%d/%m/%y"))
page_heading = BRANCH["heading"].format(date=date.today().strftime("%d/%m/%y"))

today_str = date.today().strftime("%d/%m/%y")
axis_dim_var_by_apda_json = json.dumps(AXIS_DIM_VAR_BY_APDA, ensure_ascii=False)
var_glossary_json = json.dumps(VAR_GLOSSARY_JS, ensure_ascii=False)
dim_color_json = json.dumps(DIM_COLOR_JS, ensure_ascii=False)
var_dimension_json = json.dumps(VAR_DIMENSION_JS, ensure_ascii=False)
var_label_json = json.dumps(VAR_LABEL_JS, ensure_ascii=False)

def render_page(buttons_html, sections_html, plotly_head, chrome=True, title=None, heading=None, nav_html=""):
    """Un documento HTML completo. chrome=True (index.html) incluye el
    header oscuro y la barra de tabs; chrome=False con nav_html (Dashboard/site/)
    arranca con un breadcrumb liviano (ver _site_nav); chrome=False sin nav_html
    (Dashboard/BI testing/) arranca directo en las secciones, para embeber
    dentro de Power BI, con un titulo liviano (heading) en vez del header
    oscuro. title/heading caen por defecto a page_title/page_heading (los del
    index.html). CSS compartido en Utils/report_html.py::DASHBOARD_CSS."""
    title = title or page_title
    if chrome:
        chrome_html = f"""<header>
  <h1>{page_heading}</h1>
  <p>Visualización preliminar de resultados, no compartir ni descargar figuras.</p>
</header>
<div class="tabs">
  {buttons_html}
</div>
{group_row_html}"""
    elif nav_html:
        chrome_html = nav_html
    else:
        chrome_html = f'<div class="bi-heading">{heading}</div>' if heading else ""
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{DASHBOARD_CSS}
</style>
{plotly_head}
</head>
<body>
{chrome_html}
<script>
  var VAR_DATA = {{}};
  var TARGET_SOLVE = {{}};
  var DIM_SCORES_BY_APDA = {{}};
  var AXIS_WEIGHTS = {{}};
  var TARGET_STALE = {{}};
  var AXIS_DIM_VAR_BY_APDA = {axis_dim_var_by_apda_json};
  var VAR_GLOSSARY = {var_glossary_json};
  var DIM_COLOR = {dim_color_json};
  var VAR_DIMENSION = {var_dimension_json};
  var VAR_LABEL = {var_label_json};
  var PARENT_GROUPS = {parent_groups_json};
</script>
{sections_html}
<script>
function showTab(tabId, btn) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tabs .tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  btn.classList.add('active');
  // Inicio y Decisiones no pertenecen a ningun grupo padre: se esconde la
  // segunda fila para no dejar un hijo marcado sin padre activo.
  const groupRow = document.getElementById('group-row');
  if (groupRow) groupRow.style.display = 'none';
  window.dispatchEvent(new Event('resize'));
}}

// Barra de dos niveles (solo cuando PARENT_GROUPS trae algo). showParent
// reconstruye la fila de abajo con los grupos del padre elegido y abre el
// primero; showArea alterna que seccion se ve, igual que showTab.
function showParent(parent, btn) {{
  document.querySelectorAll('.tabs .tab-btn').forEach(el => el.classList.remove('active'));
  btn.classList.add('active');
  const row = document.getElementById('group-row');
  row.querySelectorAll('.tab-btn').forEach(el => el.remove());
  (PARENT_GROUPS[parent] || []).forEach(function(pair) {{
    const b = document.createElement('button');
    b.className = 'tab-btn';
    b.textContent = pair[0];
    b.onclick = function() {{ showArea('tab-' + pair[1], b); }};
    row.appendChild(b);
  }});
  row.style.display = 'flex';
  const first = row.querySelector('.tab-btn');
  if (first) first.click();
}}
function showArea(tabId, btn) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('#group-row .tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  btn.classList.add('active');
  window.dispatchEvent(new Event('resize'));
}}
function showSubTab(groupId, tabId, btn) {{
  const group = document.getElementById(groupId);
  const parent = group.parentElement;
  parent.querySelectorAll('.subtab-content').forEach(el => el.classList.remove('active'));
  group.querySelectorAll('.subtab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  btn.classList.add('active');
  window.dispatchEvent(new Event('resize'));
}}

// Info-icon tooltips default to position:absolute inside their own
// container, so a scrollable ancestor (e.g. .target-result's overflow-x:
// auto, which per the CSS spec also makes overflow-y compute to auto)
// clips them. On hover, promote the tooltip box to position:fixed with a
// viewport-computed offset so it always renders on top, unclipped.
document.addEventListener('mouseover', function(evt) {{
  const icon = evt.target.closest('.info-icon');
  if (!icon) return;
  const box = icon.querySelector('.tooltip-box');
  if (!box) return;
  const rect = icon.getBoundingClientRect();
  const boxWidth = box.offsetWidth || 300;
  box.style.position = 'fixed';
  box.style.zIndex = 99999;
  box.style.left = Math.max(4, Math.min(rect.left, window.innerWidth - boxWidth - 8)) + 'px';
  box.style.top = 'auto';
  box.style.bottom = (window.innerHeight - rect.top + 6) + 'px';
}});

function populateSelect(sel, items) {{
  sel.innerHTML = '';
  items.forEach(it => {{
    const value = (typeof it === 'string') ? it : it.value;
    const label = (typeof it === 'string') ? it : it.label;
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    sel.appendChild(opt);
  }});
}}

var VAR_SCALE = {{}};

function initVarSelector(safe) {{
  VAR_SCALE[safe] = 'norm';
  const axisDimVar = AXIS_DIM_VAR_BY_APDA[safe];
  const axisItems = Object.keys(axisDimVar).map(k => ({{value: k, label: axisDimVar[k].label}}));
  populateSelect(document.getElementById('sel-axis-' + safe), axisItems);
  onAxisChange(safe);
}}

function onAxisChange(safe) {{
  const axisKey = document.getElementById('sel-axis-' + safe).value;
  const dims = Object.keys(AXIS_DIM_VAR_BY_APDA[safe][axisKey].dims);
  populateSelect(document.getElementById('sel-dim-' + safe), dims);
  onDimChange(safe);
}}

function onDimChange(safe) {{
  const axisKey = document.getElementById('sel-axis-' + safe).value;
  const dimName = document.getElementById('sel-dim-' + safe).value;
  const vars = AXIS_DIM_VAR_BY_APDA[safe][axisKey].dims[dimName] || [];
  const varItems = vars.map(v => ({{value: v, label: VAR_LABEL[v] || v}}));
  populateSelect(document.getElementById('sel-var-' + safe), varItems);
  onVarChange(safe);
}}

function onVarChange(safe) {{
  renderVarBar(safe);
}}

function onScaleChange(safe, mode, btn) {{
  VAR_SCALE[safe] = mode;
  const group = document.getElementById('scaletoggle-' + safe);
  group.querySelectorAll('.subtab-btn').forEach(el => el.classList.remove('active'));
  btn.classList.add('active');
  renderVarBar(safe);
}}

function updateVarInfo(safe, variable) {{
  const box = document.getElementById('varinfo-' + safe);
  if (!box) return;
  const info = VAR_GLOSSARY[variable];
  box.innerHTML = info
    ? `<b>${{info.nombre}}</b><br>${{info.descripcion}}<br><br>`
      + `<i>Cálculo:</i> ${{info.calculo}}<br>`
      + `<i>Fuente:</i> ${{info.fuente}}<br>`
      + `<i>Unidad:</i> ${{info.unidad}} · ${{info.orientacion_label}}`
    : 'Sin descripción disponible para esta variable.';
}}

function renderVarBar(safe) {{
  const variable = document.getElementById('sel-var-' + safe).value;
  updateVarInfo(safe, variable);
  const data = (VAR_DATA[safe] || {{}})[variable];
  const mode = VAR_SCALE[safe] || 'norm';
  const divId = 'varbar-' + safe;
  if (!data) {{
    Plotly.restyle(divId, {{x: [[]], y: [[]], 'marker.color': [[]], hovertext: [[]]}}, [0]);
    Plotly.relayout(divId, {{
      'title.text': (VAR_LABEL[variable] || variable) + ' — sin datos para {GROUP_THIS}',
      'xaxis.tickvals': [], 'xaxis.ticktext': [],
    }});
    return;
  }}
  const y = mode === 'raw' ? data.y_raw : data.y_norm;
  const hovertext = mode === 'raw' ? data.hovertext_raw : data.hovertext_norm;
  Plotly.restyle(divId, {{
    x: [data.x], y: [y], 'marker.color': [data.colors], hovertext: [hovertext],
  }}, [0]);
  Plotly.relayout(divId, {{
    'title.text': data.title, 'xaxis.tickvals': data.x, 'xaxis.ticktext': data.ticktext,
    'yaxis.title.text': mode === 'raw' ? 'Valor original' : 'Score normalizado (0-100)',
    'yaxis.autorange': mode === 'raw' ? true : false,
    'yaxis.range': mode === 'raw' ? null : [0, 100],
  }});
}}

function invQuantile(sortedArr, pct) {{
  if (!sortedArr || sortedArr.length === 0) return NaN;
  const n = sortedArr.length;
  if (n === 1) return sortedArr[0];
  const pos = Math.min(Math.max(pct / 100, 0), 1) * (n - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return sortedArr[lo];
  const frac = pos - lo;
  return sortedArr[lo] * (1 - frac) + sortedArr[hi] * frac;
}}

function renderTargetResult(safe, rows) {{
  const box = document.getElementById('target-result-' + safe);
  if (!box) return;
  if (!rows.length) {{ box.innerHTML = ''; return; }}
  const maxAbsPct = Math.max(...rows.map(r => Math.abs(r.pct) || 0), 1e-9);
  const trs = rows.map(r => {{
    const arrow = r.dRaw >= 0 ? '&#8593;' : '&#8595;';
    const barWidth = isNaN(r.pct) ? 0 : Math.abs(r.pct) / maxAbsPct * 100;
    const pctLabel = isNaN(r.pct) ? 'n/d' : (r.pct >= 0 ? '+' : '') + r.pct.toFixed(0) + '%';
    const info = r.info;
    const tooltip = info
      ? '<b>' + info.nombre + '</b><br>' + info.descripcion + '<br><br>'
        + '<i>Cálculo:</i> ' + info.calculo + '<br>'
        + '<i>Fuente:</i> ' + info.fuente + '<br>'
        + '<i>Unidad:</i> ' + info.unidad + ' · ' + info.orientacion_label
      : '';
    const varCell = r.label + (tooltip
      ? ' <span class="info-icon">&#128712;<span class="tooltip-box">' + tooltip + '</span></span>'
      : '');
    return '<tr>'
      + '<td><span class="dim-badge" style="background:' + r.color + '"></span>' + r.dim + '</td>'
      + '<td>' + varCell + '</td>'
      + '<td>' + r.current.toFixed(2) + '</td>'
      + '<td>' + arrow + ' ' + r.target.toFixed(2) + '</td>'
      + '<td>' + r.dz.toFixed(2) + '</td>'
      + '<td><div class="chg-cell"><div class="chg-bar-track"><div class="chg-bar" style="width:'
        + barWidth + '%;background:' + r.color + '"></div></div><span class="chg-pct">' + pctLabel + '</span></div></td>'
      + '</tr>';
  }}).join('');
  box.innerHTML = '<table class="dim-table"><thead><tr><th>Dimensión</th><th>Variable</th><th>Valor actual</th>'
    + '<th>Valor objetivo</th><th>&Delta; (desv. estándar)</th><th>Cambio aprox. (%)</th></tr></thead><tbody>' + trs + '</tbody></table>';
}}

function toggleWeightsPanel(safe) {{
  const panel = document.getElementById('weights-panel-' + safe);
  if (!panel) return;
  panel.style.display = panel.style.display === 'none' ? 'flex' : 'none';
}}

// Matches pandas Series.rank(method='average', pct=True): NaN stays NaN and
// is excluded from both the ranking and the count used for pct.
function pctRank(values) {{
  const validIdx = [];
  for (let i = 0; i < values.length; i++) {{
    if (values[i] !== null && !isNaN(values[i])) validIdx.push(i);
  }}
  const sorted = validIdx.slice().sort((a, b) => values[a] - values[b]);
  const n = sorted.length;
  const ranks = new Array(values.length).fill(NaN);
  let i = 0;
  while (i < n) {{
    let j = i;
    while (j + 1 < n && values[sorted[j + 1]] === values[sorted[i]]) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[sorted[k]] = avgRank;
    i = j + 1;
  }}
  return ranks.map(r => isNaN(r) ? NaN : (r / n) * 100);
}}

// NaN-aware weighted mean, matching Utils/data_prep.py::build_axis.
function axisValue(rowDims, weights) {{
  let wSum = 0, weighted = 0;
  for (const dim in weights) {{
    const v = rowDims[dim];
    if (v === null || v === undefined) continue;
    const w = weights[dim];
    wSum += w;
    weighted += w * v;
  }}
  return wSum > 0 ? weighted / wSum : NaN;
}}

function readWeightInputs(safe, axisKey) {{
  const table = document.getElementById('weights-table-' + axisKey + '-' + safe);
  const weights = {{}};
  let total = 0;
  table.querySelectorAll('input[data-dim]').forEach(inp => {{
    const v = parseFloat(inp.value);
    if (!isNaN(v) && v > 0) {{ weights[inp.dataset.dim] = v; total += v; }}
  }});
  const totalBox = document.getElementById('weights-total-' + axisKey + '-' + safe);
  if (totalBox) {{
    totalBox.textContent = 'Total: ' + total.toFixed(1) + ' %'
      + (Math.abs(total - 100) > 0.5 ? '  (se normaliza automáticamente)' : '');
  }}
  return weights;
}}

function recomputeGroupedPct(safe, axisKey) {{
  const data = DIM_SCORES_BY_APDA[safe];
  const weights = readWeightInputs(safe, axisKey);
  const vals = data.rows.map(r => axisValue(r, weights));
  const pct = pctRank(vals);
  const groups = [];
  let idx = 0;
  for (const size of data.group_sizes) {{
    groups.push(pct.slice(idx, idx + size));
    idx += size;
  }}
  return groups;
}}

function updateAxisWeights(safe, axisKey) {{
  if (!DIM_SCORES_BY_APDA[safe]) return;
  const groups = recomputeGroupedPct(safe, axisKey);
  const prop = axisKey === 'x' ? 'x' : 'y';
  ['vecmap-', 'mapinteractivo-'].forEach(prefix => {{
    const divId = prefix + safe;
    const div = document.getElementById(divId);
    if (!div) return;
    const update = {{}};
    update[prop] = groups;
    Plotly.restyle(divId, update, groups.map((_, i) => i));
  }});
  clearTargetResult(safe);
}}

function initWeightsTotals(safe) {{
  readWeightInputs(safe, 'x');
  readWeightInputs(safe, 'y');
}}

function resetAxisWeights(safe) {{
  const defaults = AXIS_WEIGHTS[safe];
  if (!defaults) return;
  ['x', 'y'].forEach(axisKey => {{
    const table = document.getElementById('weights-table-' + axisKey + '-' + safe);
    table.querySelectorAll('input[data-dim]').forEach(inp => {{
      inp.value = defaults[axisKey][inp.dataset.dim];
    }});
    updateAxisWeights(safe, axisKey);
  }});
  TARGET_STALE[safe] = false;
  const note = document.getElementById('target-stale-' + safe);
  if (note) note.style.display = 'none';
}}

function clearTargetResult(safe) {{
  TARGET_STALE[safe] = true;
  const note = document.getElementById('target-stale-' + safe);
  if (note) note.style.display = 'block';
  const box = document.getElementById('target-result-' + safe);
  if (box) box.innerHTML = '';
  const divId = 'mapinteractivo-' + safe;
  const div = document.getElementById(divId);
  if (!div) return;
  if (TARGET_CLICK_TRACE[safe] !== undefined) {{
    Plotly.restyle(divId, {{ x: [[]], y: [[]] }}, [TARGET_CLICK_TRACE[safe]]);
  }}
  Plotly.relayout(divId, {{ annotations: (TARGET_BASE_ANNOTATIONS[safe] || []) }});
}}

var TARGET_CLICK_TRACE = {{}};
var TARGET_BASE_ANNOTATIONS = {{}};

function handleMapTarget(safe, xClick, yClick) {{
  if (TARGET_STALE[safe]) return;
  const sol = TARGET_SOLVE[safe];
  if (!sol) return;
  const xTarget = invQuantile(sol.sorted_x, xClick);
  const yTarget = invQuantile(sol.sorted_y, yClick);
  if (isNaN(xTarget) || isNaN(yTarget)) return;
  const dAxisX = xTarget - sol.orig_axis[0];
  const dAxisY = yTarget - sol.orig_axis[1];

  const rows = sol.vars.map((v, idx) => {{
    const dz = sol.M[idx][0] * dAxisX + sol.M[idx][1] * dAxisY;
    const dRaw = dz * sol.raw_std[v];
    const current = sol.raw_current[v];
    const info = VAR_GLOSSARY[v];
    const dim = VAR_DIMENSION[v] || 'Otro';
    return {{
      label: VAR_LABEL[v] || v,
      info: info,
      dim: dim, color: DIM_COLOR[dim] || '#888',
      pct: current !== 0 ? (dRaw / current) * 100 : NaN,
      current: current, target: current + dRaw, dRaw: dRaw, dz: dz,
    }};
  }});
  const movers = rows.filter(r => Math.abs(r.dz) > 0.01);
  movers.sort((a, b) => Math.abs(b.dz) - Math.abs(a.dz));
  renderTargetResult(safe, movers.slice(0, 12));

  const divId = 'mapinteractivo-' + safe;
  const div = document.getElementById(divId);
  if (!div) return;

  // Red dot at the clicked point: a dedicated marker trace, added once and
  // repositioned via restyle on later clicks (so it always renders on top,
  // at a fixed pixel size, instead of a data-unit shape that would scale
  // oddly if the axis range ever changed).
  if (TARGET_CLICK_TRACE[safe] === undefined) {{
    TARGET_CLICK_TRACE[safe] = div.data.length;
    Plotly.addTraces(divId, {{
      x: [xClick], y: [yClick], mode: 'markers', type: 'scatter',
      marker: {{ color: '#d62828', size: 14, symbol: 'circle', line: {{ color: 'white', width: 1.5 }} }},
      name: 'Objetivo', hoverinfo: 'skip', showlegend: false,
    }});
  }} else {{
    Plotly.restyle(divId, {{ x: [[xClick]], y: [[yClick]] }}, [TARGET_CLICK_TRACE[safe]]);
  }}

  // Red arrow from UFT's current position to the clicked target. Annotations
  // are relayout'd wholesale, so we replace only our own arrow each click and
  // keep whatever base annotations (e.g. the watermark) were there at init.
  const arrow = {{
    x: xClick, y: yClick, ax: sol.uft_plot[0], ay: sol.uft_plot[1],
    xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
    showarrow: true, arrowhead: 3, arrowsize: 1.4, arrowwidth: 2.5,
    arrowcolor: '#d62828', text: '',
  }};
  Plotly.relayout(divId, {{
    annotations: (TARGET_BASE_ANNOTATIONS[safe] || []).concat([arrow]),
  }});
}}

function initTargetFinder(safe) {{
  const divId = 'mapinteractivo-' + safe;
  const div = document.getElementById(divId);
  if (!div || !TARGET_SOLVE[safe]) return;
  TARGET_BASE_ANNOTATIONS[safe] = (div.layout.annotations || []).slice();
  // plotly.js has no public "click anywhere in the plot area -> data coords"
  // API for cartesian traces (plotly_click only fires on marker hits), so we
  // read the current axis range + plot-area geometry off the graph div's
  // internal _fullLayout and convert the raw pixel click ourselves. Re-reads
  // .range on every click so it stays correct across zoom/pan.
  div.addEventListener('click', function(evt) {{
    const fl = div._fullLayout;
    if (!fl || !fl.xaxis || !fl.yaxis) return;
    const rect = div.getBoundingClientRect();
    const size = fl._size;
    const px = evt.clientX - rect.left - size.l;
    const py = evt.clientY - rect.top - size.t;
    if (px < 0 || py < 0 || px > size.w || py > size.h) return;
    const xData = fl.xaxis.range[0] + (px / size.w) * (fl.xaxis.range[1] - fl.xaxis.range[0]);
    const yData = fl.yaxis.range[1] - (py / size.h) * (fl.yaxis.range[1] - fl.yaxis.range[0]);
    handleMapTarget(safe, xData, yData);
  }});
}}

// Plotly captures its container's width once at first render; config's
// responsive:true only re-checks it on the browser *window*'s resize event,
// not the container's own. That's fine in a normal tab, but inside an
// embedded iframe (e.g. a Power BI visual) the container can change size —
// on first settle, or when a hidden subtab (display:none -> block) becomes
// visible — without the iframe's window ever firing 'resize'. A
// ResizeObserver per chart catches that directly and re-lays it out.
document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {{
  new ResizeObserver(function() {{ Plotly.Plots.resize(gd); }}).observe(gd);
}});
</script>
</body>
</html>
"""


plotly_js = get_plotlyjs()

html = render_page("".join(all_buttons), "".join(all_sections), f"<script>{plotly_js}</script>")
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\nSaved: {OUT_PATH}  ({len(apdas)} {GROUP_PLURAL}, {os.path.getsize(OUT_PATH)/1e6:.1f} MB)")

# ── Dashboard/BI testing/ ───────────────────────────────────────────────────
# Una pagina standalone por APDA (sin header ni barra de tabs, sin PCA / PCA +
# AA), pensada para embeberse como iframe dentro de un reporte de Power BI.
# plotly.js va una sola vez, compartido, en vez de inline en cada pagina.
# Desactivado en la rama Cine (generate_bi_pages: false en su config.yaml) —
# esas paginas se generaron por error y no estan pensadas para esa rama.
if BRANCH["generate_bi_pages"]:
    BI_DIR = os.path.join(os.path.dirname(OUT_DIR), "BI testing")
    os.makedirs(BI_DIR, exist_ok=True)
    with open(f"{BI_DIR}/plotly.min.js", "w", encoding="utf-8") as f:
        f.write(plotly_js)
    for apda, section in bi_sections.items():
        bi_html = render_page("", section, '<script src="plotly.min.js"></script>', chrome=False,
                               title=f"{apda} — Dashboard {GROUP_PLURAL}", heading=apda)
        with open(f"{BI_DIR}/{safe_id(apda)}.html", "w", encoding="utf-8") as f:
            f.write(bi_html)
    print(f"Saved: {BI_DIR}  ({len(bi_sections)} {GROUP_PLURAL} + plotly.min.js, "
          f"{os.path.getsize(f'{BI_DIR}/plotly.min.js')/1e6:.1f} MB)")


# =============================================================================
# Dashboard/site/ — unified static site (Part A). Both branch runs (default
# APDA, and `python build_dashboard.py Cine/config.yaml`) reach this block;
# each writes its own group's pages plus the fragments/assets the hub
# (Dashboard/build_site.py) assembles afterwards. Reuses render_page's CSS/JS
# wholesale (chrome=False, same mechanism as Dashboard/BI testing/) — the only
# new piece is nav_html, a breadcrumb bar built by _site_nav.
# =============================================================================
SITE_ASSETS_DIR = "site/assets"
os.makedirs(SITE_ASSETS_DIR, exist_ok=True)
with open(f"{SITE_ASSETS_DIR}/plotly.min.js", "w", encoding="utf-8") as f:
    f.write(plotly_js)

os.makedirs("site/_fragments", exist_ok=True)
if PARENT_GROUPS:
    with open("site/_fragments/inicio-cine.html", "w", encoding="utf-8") as f:
        f.write(intro_note_block)
else:
    with open("site/_fragments/inicio-apda.html", "w", encoding="utf-8") as f:
        f.write(welcome_section)
    with open("site/_fragments/decisiones.html", "w", encoding="utf-8") as f:
        f.write(decisiones_section)

SITE_DIR = BRANCH["site_dir"]
os.makedirs(SITE_DIR, exist_ok=True)
is_cine_branch = bool(PARENT_GROUPS)
for group_name, section in site_sections.items():
    site_html = render_page(
        "", section, '<script src="../assets/plotly.min.js"></script>', chrome=False,
        title=f"{group_name} — Dashboard {GROUP_PLURAL}", nav_html=_site_nav(group_name, is_cine_branch),
    )
    with open(f"{SITE_DIR}/{slug(group_name)}.html", "w", encoding="utf-8") as f:
        f.write(site_html)
print(f"Saved: {SITE_DIR}  ({len(site_sections)} {GROUP_PLURAL}, site pages)")
