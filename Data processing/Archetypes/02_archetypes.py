# =============================================================================
# Archetypes/02_archetypes.py — Archetypal analysis per APDA
# =============================================================================
# Reads Archetypes/config.yaml and Archetypes/outputs/dimension_scores.csv
# (built by Archetypes/01_dimension_scores.py) and fits archetypal analysis
# (AA, via the `archetypes` package) SEPARATELY within each of the 4 APDAs,
# on the same 9 curated 0-100 dimension scores Vector/01_vectors.py uses for
# its PCA biplots.
#
# For each APDA:
#   1. Zero-fill "Investigación ANID (APDA)" (NaN there means zero ANID-funded
#      projects, a genuine zero rather than an unknown value — see
#      ZERO_FILL_DIMS below), then median-impute + z-standardize the rest of
#      the dimension scores (same pattern as Vector's PCA step).
#   2. Fit AA for each k in config["k_candidates"] (capped to stay sane
#      relative to the ~34-35 institutions per APDA), pick k via an RSS-elbow
#      heuristic (overridable per APDA via config["k_override"]).
#   3. Save archetype profiles (inverse-transformed back to the 0-100
#      dimension-score scale) and institution membership weights (alpha).
#   4. Plot: RSS elbow, stacked-bar of memberships, 2-simplex (if k==3), and
#      a radar chart of archetype profiles — reusing Vector's UFT-highlight /
#      accreditation-year color / watermark conventions.
#
# NOTE — compatibility shim: the only Windows-installable build of the
# `archetypes` package on this machine (0.6.2 — newer releases need a C/C++
# toolchain this machine doesn't have) predates two later removals in its
# dependencies. See Utils/archetypes_compat.py for the shims (shared with
# 04_no_imputation.py, 06_stability_check.py, 07_archetypoids.py).
#
# Usage (from project root):
#   python Archetypes/00_prepare_data.py
#   python Archetypes/01_dimension_scores.py
#   python Archetypes/02_archetypes.py
# =============================================================================

# %% Setup
import os
import sys
import warnings
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3d projection
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "Utils")
from utils import abreviaciones_ies

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── archetypes/sklearn/numpy compatibility shims (see module docstring) ─────
import archetypes_compat  # noqa: F401 — patches sklearn/numpy before archetypes import
from archetypes import AA

warnings.filterwarnings("ignore")

# %% Config
# Los paths salen de argv para que otra rama del pipeline (ej. Cine/, que
# agrupa por area CINE-F 13 en vez de por APDA) reuse este script tal cual:
#   python Archetypes/02_archetypes.py <config.yaml> <out_dir>
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "Archetypes/config.yaml"
OUT_DIR     = sys.argv[2] if len(sys.argv) > 2 else "Archetypes/outputs"
IN_PATH     = f"{OUT_DIR}/dimension_scores.csv"
os.makedirs(OUT_DIR, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

TARGET_COL   = cfg["target"]
K_CANDIDATES = sorted(cfg["k_candidates"])
K_OVERRIDE   = cfg.get("k_override", {}) or {}
AA_CFG       = cfg["aa"]
PLOT_CFG     = cfg["plot"]
WATERMARK    = cfg.get("watermark", False)

# Dimensions where NaN means a genuine, defensible zero rather than an
# unknown value (see module docstring) — filled with 0 before imputing the
# rest, instead of being median-imputed like every other dimension.
ZERO_FILL_DIMS = ["Investigación ANID (APDA)", "Producción Científica (APDA)"]

if not os.path.exists(IN_PATH):
    print(f"Not found: {IN_PATH}\nCorre 01_dimension_scores.py con esta misma config primero.")
    sys.exit(1)

dim_scores = pd.read_csv(IN_PATH, encoding="utf-8-sig")
id_cols    = [c for c in ("ies_norm", "apda") if c in dim_scores.columns]
non_dim    = set(id_cols) | {TARGET_COL}
valid_dims = [c for c in dim_scores.columns if c not in non_dim]
print(f"Loaded {len(dim_scores)} IES×APDA cells, {len(valid_dims)} dimensions: {valid_dims}")

dim_scores["ies_label"] = (
    dim_scores["ies_norm"].map(abreviaciones_ies).fillna(dim_scores["ies_norm"])
)

# %% Plot colors — accreditation years + UFT highlight (same convention as Vector/)
ACRED_YEAR_COLORS = {
    1: "#4C72B0", 2: "#DD8452", 3: "#55A868", 4: "#4C72B0",
    5: "#DD8452", 6: "#55A868", 7: "#8172B2",
}
UFT_IES_NORM = "universidad finis terrae"
UFT_COLOR    = "#00b7eb"

_year_vals = sorted(pd.to_numeric(dim_scores[TARGET_COL], errors="coerce")
                     .dropna().astype(int).unique())
_fallback  = plt.cm.get_cmap("tab10", max(len(_year_vals), 1))
COLOR_MAP_QUAL = {v: ACRED_YEAR_COLORS.get(v, _fallback(i)) for i, v in enumerate(_year_vals)}
dim_scores["_is_uft"] = dim_scores["ies_norm"].eq(UFT_IES_NORM)


def _add_watermark(ax):
    ax.text(0.5, 0.5, "PRELIMINAR - NO REPRODUCIR",
            transform=ax.transAxes, fontsize=22, color="gray",
            alpha=0.18, ha="center", va="center", rotation=30,
            fontweight="bold", bbox=dict(boxstyle="round", fc="none", ec="none"))


def _point_colors(sub):
    yrs = pd.to_numeric(sub[TARGET_COL], errors="coerce")
    colors = [COLOR_MAP_QUAL.get(int(v), "grey") if pd.notna(v) else "grey" for v in yrs]
    return [UFT_COLOR if is_uft else c for is_uft, c in zip(sub["_is_uft"], colors)]


# %% Elbow (RSS) k-selection
def _pick_k_elbow(ks, rss_vals):
    """Max-distance-to-chord ('kneedle') elbow pick over (k, RSS) pairs."""
    ks, rss = np.array(ks, dtype=float), np.array(rss_vals, dtype=float)
    if len(ks) == 1:
        return int(ks[0])
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (rss - rss.min()) / (rss.max() - rss.min() + 1e-12)
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)
    return int(ks[np.argmax(dist)])


# %% Per-APDA archetype fitting
memberships_rows = []
apdas = sorted(dim_scores["apda"].dropna().unique())

for apda in apdas:
    print(f"\n{'=' * 70}\nAPDA: {apda}\n{'=' * 70}")
    sub = dim_scores[dim_scores["apda"] == apda].reset_index(drop=True)
    n = len(sub)
    safe = apda.replace("/", "-").replace(" ", "_")

    # a dimension can be entirely NaN within this APDA's rows (e.g. a
    # dimension scoped to a single other APDA, like "Acreditación
    # Trayectoria" outside Salud y bienestar); SimpleImputer silently drops
    # such columns, so exclude them upfront to keep local_dims aligned with
    # the fitted archetype coordinates below.
    local_dims = [d for d in valid_dims if sub[d].notna().any()]
    X_pre = sub[local_dims].copy()
    for d in ZERO_FILL_DIMS:
        if d in X_pre.columns:
            X_pre[d] = X_pre[d].fillna(0.0)
    X_imp = SimpleImputer(strategy="median").fit_transform(X_pre)
    scaler = StandardScaler().fit(X_imp)
    X_std = scaler.transform(X_imp)

    # Cap candidate k relative to N — archetypes are extreme corner points, so
    # with N this small a k too close to N risks single-institution archetypes.
    k_candidates = [k for k in K_CANDIDATES if k < n - 1]
    if not k_candidates:
        print(f"  Skipping: too few institutions ({n}) for any candidate k.")
        continue
    if max(k_candidates) > n / 6:
        print(f"  Warning: N={n} is thin relative to k up to {max(k_candidates)} "
              f"— sanity-check archetypes against raw institution profiles before naming them.")

    fits = {}
    for k in k_candidates:
        aa = AA(n_archetypes=k, n_init=AA_CFG["n_init"], max_iter=AA_CFG["max_iter"],
                 tol=AA_CFG["tol"], random_state=AA_CFG["random_state"])
        aa.fit(X_std)
        fits[k] = aa
        print(f"  k={k}  RSS={aa.rss_:.3f}")

    rss_table = pd.DataFrame({"k": k_candidates, "rss": [fits[k].rss_ for k in k_candidates]})
    rss_table.to_csv(f"{OUT_DIR}/k_selection_{safe}.csv", index=False, encoding="utf-8-sig")

    chosen_k = K_OVERRIDE.get(apda, None) or _pick_k_elbow(k_candidates, rss_table["rss"].values)
    print(f"  -> chosen k = {chosen_k}"
          f"{' (override)' if apda in K_OVERRIDE else ' (elbow)'}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(rss_table["k"], rss_table["rss"], "o-", color="steelblue")
    ax.axvline(chosen_k, color="crimson", linestyle="--", label=f"chosen k={chosen_k}")
    ax.set_xlabel("n_archetypes (k)")
    ax.set_ylabel("RSS")
    ax.set_title(f"Selección de k — APDA: {apda}  (N={n})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    aa = fits[chosen_k]

    # Archetype profiles, back on the familiar 0-100 dimension-score scale.
    archetype_coords = scaler.inverse_transform(aa.archetypes_)
    profiles = pd.DataFrame(archetype_coords, columns=local_dims)
    profiles.insert(0, "archetype", [f"A{i+1}" for i in range(chosen_k)])
    profiles.insert(0, "apda", apda)
    profiles.to_csv(f"{OUT_DIR}/archetype_profiles_{safe}.csv", index=False, encoding="utf-8-sig")

    # Institution memberships (alpha weights, row-stochastic).
    alphas = aa.alphas_
    dominant = alphas.argmax(axis=1)
    mem = sub[id_cols + ["ies_label", "_is_uft", TARGET_COL]].copy()
    for i in range(chosen_k):
        mem[f"alpha_{i+1}"] = alphas[:, i]
    mem["dominant_archetype"] = [f"A{i+1}" for i in dominant]
    mem["dominant_share"] = alphas.max(axis=1)
    mem["k"] = chosen_k
    memberships_rows.append(mem)

    # ── Plot: stacked bar of memberships, sorted by dominant archetype/share ──
    order = mem.sort_values(["dominant_archetype", "dominant_share"], ascending=[True, False]).index
    mem_sorted = mem.loc[order].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=tuple(PLOT_CFG["figsize"]))
    bottom = np.zeros(len(mem_sorted))
    arch_colors = plt.cm.get_cmap("Set2", chosen_k)
    for i in range(chosen_k):
        vals = mem_sorted[f"alpha_{i+1}"].values
        ax.bar(range(len(mem_sorted)), vals, bottom=bottom, color=arch_colors(i),
               label=f"A{i+1}", width=0.85)
        bottom += vals
    ax.set_xticks(range(len(mem_sorted)))
    ax.set_xticklabels(mem_sorted["ies_label"], rotation=90, fontsize=7)
    for tick, is_uft in zip(ax.get_xticklabels(), mem_sorted["_is_uft"]):
        if is_uft:
            tick.set_color(UFT_COLOR)
            tick.set_fontweight("bold")
    ax.set_ylabel("Peso de membresía (alpha)")
    ax.set_title(f"Membresía a arquetipos — APDA: {apda}  (k={chosen_k})")
    ax.legend(title="Arquetipo", loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.02)
    if WATERMARK:
        _add_watermark(ax)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # ── Plot: 2-simplex (ternary) — only meaningful/readable when k == 3 ──────
    if chosen_k == 3:
        a1, a2, a3 = alphas[:, 0], alphas[:, 1], alphas[:, 2]
        x = a2 + 0.5 * a3
        y = (np.sqrt(3) / 2) * a3
        colors = _point_colors(sub)

        fig, ax = plt.subplots(figsize=tuple(PLOT_CFG["figsize"]))
        tri_x, tri_y = [0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0]
        ax.plot(tri_x, tri_y, color="#999999", linewidth=1)
        ax.scatter(x, y, color=colors, s=90, alpha=0.9, edgecolors="white",
                   linewidths=0.6, zorder=3)
        for xi, yi, is_uft, label in zip(x, y, sub["_is_uft"], sub["ies_label"]):
            ax.annotate(label, (xi, yi), fontsize=9 if is_uft else 7,
                        alpha=0.95 if is_uft else 0.8,
                        color=UFT_COLOR if is_uft else "black",
                        textcoords="offset points", xytext=(5, 4),
                        fontweight="bold" if is_uft else "normal")
        for vx, vy, label in zip([0, 1, 0.5], [0, 0, np.sqrt(3) / 2], ["A1", "A2", "A3"]):
            ax.annotate(label, (vx, vy), fontsize=11, fontweight="bold", color="crimson",
                        ha="center", va="bottom" if vy > 0 else "top",
                        xytext=(0, 8 if vy > 0 else -12), textcoords="offset points")
        _present = sorted(pd.to_numeric(sub[TARGET_COL], errors="coerce")
                           .dropna().astype(int).unique())
        handles = [Patch(color=COLOR_MAP_QUAL[v], label=f"{v} años") for v in _present]
        if sub["_is_uft"].any():
            handles.append(Patch(color=UFT_COLOR, label="UFT"))
        ax.legend(handles=handles, title="Años acreditación", fontsize=8,
                  title_fontsize=8, loc="upper right")
        ax.set_title(f"Simplex de arquetipos — APDA: {apda}")
        ax.set_aspect("equal")
        ax.axis("off")
        if WATERMARK:
            _add_watermark(ax)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # ── Plot: 3-simplex (tetrahedron) — exact geometric analog for k == 4 ─────
    # A k==3 simplex is a triangle (2D); a k==4 simplex is a tetrahedron (3D).
    # Same barycentric idea as the ternary plot above: each institution's
    # position is alphas @ vertices, so it sits exactly where its membership
    # mix places it, no PCA approximation involved (contrast with
    # the dashboard's interactive PCA map, which handles any k but is only an
    # approximate 2D shadow for k > 3).
    elif chosen_k == 4:
        tetra_verts = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
        ], dtype=float)
        points = alphas @ tetra_verts
        colors = _point_colors(sub)

        fig = plt.figure(figsize=tuple(PLOT_CFG["figsize"]))
        ax = fig.add_subplot(111, projection="3d")
        for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            ax.plot(*zip(tetra_verts[i], tetra_verts[j]), color="#999999", linewidth=1)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors, s=80,
                   alpha=0.9, edgecolors="white", linewidths=0.6, depthshade=True)
        for xi, yi, zi, is_uft, label in zip(points[:, 0], points[:, 1], points[:, 2],
                                              sub["_is_uft"], sub["ies_label"]):
            ax.text(xi, yi, zi, label, fontsize=9 if is_uft else 7,
                    alpha=0.95 if is_uft else 0.8,
                    color=UFT_COLOR if is_uft else "black",
                    fontweight="bold" if is_uft else "normal")
        for (vx, vy, vz), label in zip(tetra_verts * 1.2, ["A1", "A2", "A3", "A4"]):
            ax.text(vx, vy, vz, label, fontsize=12, fontweight="bold", color="crimson", ha="center")
        _present = sorted(pd.to_numeric(sub[TARGET_COL], errors="coerce")
                           .dropna().astype(int).unique())
        handles = [Patch(color=COLOR_MAP_QUAL[v], label=f"{v} años") for v in _present]
        if sub["_is_uft"].any():
            handles.append(Patch(color=UFT_COLOR, label="UFT"))
        ax.legend(handles=handles, title="Años acreditación", fontsize=8,
                  title_fontsize=8, loc="upper right")
        ax.set_title(f"Simplex de arquetipos (tetraedro) — APDA: {apda}")
        ax.view_init(elev=20, azim=35)
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass  # older matplotlib without set_box_aspect
        ax.set_axis_off()
        if WATERMARK:
            # 3D axes need text2D for a flat overlay — ax.text() there takes
            # (x, y, z, s, ...), not the 2D _add_watermark(ax) signature.
            ax.text2D(0.5, 0.5, "PRELIMINAR - NO REPRODUCIR",
                      transform=ax.transAxes, fontsize=22, color="gray",
                      alpha=0.18, ha="center", va="center", rotation=30,
                      fontweight="bold", bbox=dict(boxstyle="round", fc="none", ec="none"))
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # ── Plot: radar chart of archetype profiles ───────────────────────────────
    n_ax = len(local_dims)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=tuple(PLOT_CFG["figsize"]), subplot_kw=dict(polar=True))
    for i in range(chosen_k):
        vals = archetype_coords[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, marker="o", label=f"A{i+1}", color=arch_colors(i))
        ax.fill(angles, vals, alpha=0.08, color=arch_colors(i))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(local_dims, fontsize=8)
    ax.set_title(f"Perfil de arquetipos — APDA: {apda}", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    if WATERMARK:
        _add_watermark(ax)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    print(f"  Saved: archetype_profiles_{safe}.csv, k_selection_{safe}.csv")

# %% Combined memberships output
memberships = pd.concat(memberships_rows, ignore_index=True)
memberships.to_csv(f"{OUT_DIR}/archetype_memberships.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUT_DIR}/archetype_memberships.csv  ({len(memberships)} institutions×APDA)")
