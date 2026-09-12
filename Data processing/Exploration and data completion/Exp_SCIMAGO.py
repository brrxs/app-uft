# %% Setup

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

exec(open('Utils/utils.py', encoding='utf-8').read())

# %% Load

scimago = pd.read_csv('datos/Scimago/chilean_universities_subject_ranks.csv')


col_info = pd.DataFrame({
    'dtype':         [str(scimago[c].dtype) for c in scimago.columns],
    'n_no_null':     [int(scimago[c].notna().sum()) for c in scimago.columns],
    'cobertura_pct': [round(scimago[c].notna().mean() * 100, 1) for c in scimago.columns],
    'n_unique':      [scimago[c].nunique() for c in scimago.columns],
})
col_info.index = scimago.columns

print(col_info.to_string())

# %% cleaning

mask_univ = (
    scimago['institution'].str.contains('Universidad', na=False) &
    ~scimago['institution'].str.contains('Hospital', na=False)&
    scimago['scope'].str.contains('Chile', na=False)
)

scimago_univ = scimago[mask_univ].copy()

print(f"{scimago_univ['institution'].nunique()} universidades — {scimago_univ.shape[0]:,} filas")
print(scimago_univ['institution'].unique())

print(scimago_univ.head(10))

# %% Pivot: universities × subjects

scimago_wide = scimago_univ.pivot(
    index='institution',
    columns='area_name',
    values='rank_value',
)
scimago_wide.columns.name = None
scimago_wide.index.name = 'institution'

print(scimago_wide.shape)
print(scimago_wide.head())

# %% Export

scimago_wide.to_excel('Data/catalogos/scimago_wide.xlsx')

# =============================================================================
# APDA SCORING FROM SCIMAGO SUBJECT RANKS
# -----------------------------------------------------------------------------
# Criterion documentation
# -----------------------
# rank_value is a raw ordinal rank (1 = top, higher = worse). Pool sizes
# differ by subject (not all 43 universities appear in every subject area), so
# raw ranks cannot be averaged directly across subjects.
#
# Two scoring tracks are computed and exported:
#
#   Option C (weighted)  — pct_score_w = mean_pct × coverage_ratio
#      Absent institutions = 0. Penalises partial coverage across subjects.
#
#   Presence-only        — mean(pct_score) across ranked subjects only.
#      Absent institutions = NaN (excluded from each APDA's pool).
#      Avoids conflating absence with poor performance.
#
# pct_score = (n_s - rank + 1) / n_s * 100  — scale-free [0–100], higher = better.
# Aggregation: mean of subject scores where institution appears; subjects spanning
# multiple APDAs contribute to each APDA independently.
# =============================================================================

# %% Subject → APDA crosswalk

SUBJECT_TO_APDA = {
    "Agricultural and Biological Sciences":         ["Ciencias, tecnología y calidad de vida"],
    "Arts and Humanities":                          ["Artes, humanidades y cultura"],
    "Biochemistry, Genetics and Molecular Biology": ["Ciencias, tecnología y calidad de vida"],
    "Business, Management and Accounting":          ["Ed., Sociedad y desarrollo humano"],
    "Chemistry":                                    ["Ciencias, tecnología y calidad de vida"],
    "Computer Science":                             ["Ciencias, tecnología y calidad de vida"],
    "Dentistry":                                    ["Salud y bienestar"],
    "Earth and Planetary Sciences":                 ["Ciencias, tecnología y calidad de vida"],
    "Economics, Econometrics and Finance":          ["Ed., Sociedad y desarrollo humano"],
    "Energy":                                       ["Ciencias, tecnología y calidad de vida"],
    "Engineering":                                  ["Ciencias, tecnología y calidad de vida"],
    "Environmental Science":                        ["Ciencias, tecnología y calidad de vida"],
    "Mathematics":                                  ["Ciencias, tecnología y calidad de vida"],
    "Medicine":                                     ["Salud y bienestar"],
    "Pharmacology, Toxicology and Pharmaceutics":   ["Salud y bienestar"],
    "Physics and Astronomy":                        ["Ciencias, tecnología y calidad de vida"],
    "Psychology":                                   ["Ed., Sociedad y desarrollo humano"],
    "Social Sciences":                              ["Ed., Sociedad y desarrollo humano"],
    # "Veterinary":                                   ["Ciencias, tecnología y calidad de vida"],
}

# %% Compute per-subject scores (three methods)

scimago_scored = scimago_univ.copy()

# Re-rank within the filtered pool (universities, Chile scope) so that rank 1
# always maps to the top institution in our dataset. Using the raw rank_value
# directly would yield negative scores because hospitals were excluded but
# their original ranks remain in rank_value.
scimago_scored['rank_filtered'] = (
    scimago_scored.groupby('area_name')['rank_value']
    .rank(method='min', ascending=True)
)

# Pool size per subject (n_s = # institutions ranked in Chile for that subject)
scimago_scored['n_subject'] = (
    scimago_scored.groupby('area_name')['rank_value'].transform('count')
)

# Method 1: relative percentile [0–100], higher = better
scimago_scored['pct_score'] = (
    (scimago_scored['n_subject'] - scimago_scored['rank_filtered'] + 1)
    / scimago_scored['n_subject'] * 100
)


# %% Explode crosswalk and aggregate to institution × APDA

crosswalk = pd.DataFrame(
    [(subj, apda) for subj, apdas in SUBJECT_TO_APDA.items() for apda in apdas],
    columns=['area_name', 'apda']
)

scimago_long = scimago_scored.merge(crosswalk, on='area_name', how='left')

apda_scores = (
    scimago_long
    .groupby(['institution', 'apda'])
    .agg(
        pct_score  = ('pct_score', 'mean'),
        n_subjects = ('area_name', 'count'),
    )
    .reset_index()
)

apda_pct = apda_scores.pivot(index='institution', columns='apda', values='pct_score')
apda_pct.columns.name = None
apda_pct = apda_pct.apply(lambda col: col / col.max() * 100)  # top IES per APDA = 100

print(f"APDA score tables: {apda_pct.shape[0]} institutions × {apda_pct.shape[1]} APDAs")

# %% Coverage penalty — Option C
# final_score = mean_pct × (n_subjects_present / n_subjects_possible_in_apda)
# Institutions with no presence in an APDA receive score 0.

n_possible = crosswalk.groupby('apda')['area_name'].count()

all_institutions = scimago_univ['institution'].unique()
APDAS_LIST       = sorted(n_possible.index.tolist())
full_index       = pd.MultiIndex.from_product(
    [all_institutions, APDAS_LIST], names=['institution', 'apda']
)

apda_full = (
    apda_scores
    .set_index(['institution', 'apda'])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

apda_full['n_possible']     = apda_full['apda'].map(n_possible)
apda_full['coverage_ratio'] = apda_full['n_subjects'] / apda_full['n_possible']
apda_full['pct_score_w']    = apda_full['pct_score'] * apda_full['coverage_ratio']

apda_pct_w = apda_full.pivot(index='institution', columns='apda', values='pct_score_w')
apda_pct_w.columns.name = None
apda_pct_w = apda_pct_w.apply(lambda col: col / col.max() * 100)  # top IES per APDA = 100

print(f"\nWeighted tables: {apda_pct_w.shape} — no NaN: {apda_pct_w.isna().sum().sum() == 0}")

# %% Diagnostics

APDAS = apda_pct.columns.tolist()

print("\n--- Coverage (% institutions with a score per APDA) ---")
for apda in APDAS:
    cov_pct = apda_pct[apda].notna().mean() * 100
    print(f"  {apda:<45} pct={cov_pct:.0f}%")

print("\n--- Top 5 per APDA (percentile method) ---")
for apda in APDAS:
    top5 = apda_pct[apda].dropna().sort_values(ascending=False).head(5)
    print(f"\n  {apda}:")
    for inst, val in top5.items():
        print(f"    {inst:<50} {val:.1f}")

# %% Visualización: una figura por APDA (percentile vs inv_rank normalizado)

def _shorten(name):
    return (name
        .replace('Pontificia Universidad Católica', 'PUC')
        .replace('Pontificia Universidad Catolica', 'PUC')
        .replace('Universidad de ', 'U. ')
        .replace('Universidad del ', 'U. del ')
        .replace('Universidad Técnica ', 'U.T. ')
        .replace('Universidad Tecnológica ', 'U.T. ')
        .replace('Universidad Austral', 'U. Austral')
    )

COLOR_PCT  = '#2171b5'
COLOR_ZERO = '#cccccc'   # unranked institutions

for apda in APDAS:
    # Option C: all institutions, sorted by weighted score
    order_w  = apda_pct_w[apda].sort_values(ascending=True)
    n_w      = len(order_w)
    vals_w   = order_w.values
    labels_w = [_shorten(i) for i in order_w.index]
    colors_w = [COLOR_PCT if v > 0 else COLOR_ZERO for v in vals_w]

    # Presence-only: exclude absent institutions
    series_p  = apda_pct[apda].dropna().sort_values(ascending=True)
    n_p       = len(series_p)
    labels_p  = [_shorten(i) for i in series_p.index]

    fig, (ax_w, ax_p) = plt.subplots(
        1, 2,
        figsize=(14, max(4, max(n_w, n_p) * 0.32)),
        sharey=False,
    )

    # Left: Option C
    ax_w.barh(range(n_w), vals_w, height=0.6, color=colors_w, zorder=2)
    ax_w.set_yticks(range(n_w))
    ax_w.set_yticklabels(labels_w, fontsize=8)
    ax_w.set_xlabel('Score [0–100]', fontsize=9)
    ax_w.set_xlim(0, 110)
    ax_w.axvline(50, color='#aaa', lw=0.8, ls='--', zorder=1)
    ax_w.grid(axis='x', lw=0.4, color='#ddd', zorder=0)
    ax_w.spines[['top', 'right']].set_visible(False)
    ax_w.set_title(f'Option C — cobertura ponderada\n(n={n_w})', fontsize=9, pad=6)

    # Right: presence-only
    ax_p.barh(range(n_p), series_p.values, height=0.6, color=COLOR_PCT, zorder=2)
    ax_p.set_yticks(range(n_p))
    ax_p.set_yticklabels(labels_p, fontsize=8)
    ax_p.set_xlabel('Score [0–100]', fontsize=9)
    ax_p.set_xlim(0, 110)
    ax_p.axvline(50, color='#aaa', lw=0.8, ls='--', zorder=1)
    ax_p.grid(axis='x', lw=0.4, color='#ddd', zorder=0)
    ax_p.spines[['top', 'right']].set_visible(False)
    ax_p.set_title(f'Presence-only — solo rankeadas\n(n={n_p})', fontsize=9, pad=6)

    fig.suptitle(f'Scimago — {apda}', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()

# %% Export APDA scores

with pd.ExcelWriter('Data/catalogos/scimago_apda_scores.xlsx') as w:
    apda_pct_w.to_excel(w, sheet_name='percentile_w')
    apda_pct.to_excel(w, sheet_name='presence_only')
    apda_full.to_excel(w, sheet_name='long', index=False)


# %% Mapear institution → ies_norm y exportar CSVs

# Diccionario de correcciones para nombres que no coinciden después de normalizar
_scimago_fix = {
    'universidad andres bello, chile':  'universidad andres bello',
    'universidad de las americas, chile': 'universidad de las americas',
    'universidad de los andes, chile':  'universidad de los andes',
    'universidad de playa ancha':       'universidad de playa ancha de ciencias de la educacion',
    'universidad santo tomas, chile':   'universidad santo tomas',
}

# Asignar ies_norm a cada institución en apda_pct_w y apda_pct (indexadas por institution)
def _ies_norm_scimago(inst):
    raw = normalizar_nombre_ies(inst)
    return _scimago_fix.get(raw, raw)

# ── Long table: IES×APDA con ambas métricas ────────────────────────────────

# apda_pct_w: institution × APDA (columnas)
pct_w_long = (
    apda_pct_w
    .reset_index()
    .melt(id_vars='institution', var_name='apda', value_name='scimago_apda_score_w')
)

# apda_pct: institution × APDA (columnas), puede tener NaN para ausentes
pct_long = (
    apda_pct
    .reset_index()
    .melt(id_vars='institution', var_name='apda', value_name='scimago_apda_score_presence')
)

scimago_long_out = pct_w_long.merge(pct_long, on=['institution', 'apda'], how='left')
scimago_long_out['ies_norm'] = scimago_long_out['institution'].apply(_ies_norm_scimago)

# Diagnóstico: instituciones sin match
mf_norms = set(pd.read_csv('Data/merged_final.csv', encoding='utf-8-sig',
                            low_memory=False)['ies_norm'].unique())
_unmatched_sci = [
    inst for inst in scimago_long_out['institution'].unique()
    if _ies_norm_scimago(inst) not in mf_norms
]
print("\n── Scimago institutions sin match en merged_final ──")
if _unmatched_sci:
    for u in sorted(_unmatched_sci):
        print(f"  {u!r}  →  ies_norm: {_ies_norm_scimago(u)!r}")
else:
    print("  (ninguna)")

# Seleccionar columnas de salida
scimago_ies_apda = scimago_long_out[
    ['ies_norm', 'apda', 'scimago_apda_score_w', 'scimago_apda_score_presence']
].copy()

print(f"\nscimago_ies_apda: {scimago_ies_apda.shape[0]} filas  "
      f"({scimago_ies_apda['ies_norm'].nunique()} IES × {scimago_ies_apda['apda'].nunique()} APDAs)")

# ── IES-level rollup: media de scores por institución ─────────────────────
scimago_ies = (
    scimago_ies_apda
    .groupby('ies_norm')
    .agg(
        scimago_ies_score_w_mean        = ('scimago_apda_score_w',        'mean'),
        scimago_ies_score_presence_mean = ('scimago_apda_score_presence', 'mean'),
    )
    .reset_index()
)

print(f"scimago_ies: {scimago_ies.shape[0]} IES")

# ── Exportar ────────────────────────────────────────────────────────────────
os.makedirs('Data/catalogos', exist_ok=True)
scimago_ies_apda.to_csv('Data/catalogos/scimago_ies_apda.csv', index=False, encoding='utf-8-sig')
scimago_ies.to_csv('Data/catalogos/scimago_ies.csv',           index=False, encoding='utf-8-sig')

print("Exportado: Data/catalogos/scimago_ies_apda.csv")
print("Exportado: Data/catalogos/scimago_ies.csv")
