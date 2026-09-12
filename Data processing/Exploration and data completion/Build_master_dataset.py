# =============================================================================
# Build_master_dataset.py
# Combina merged_final.csv con datos ANID y Scimago a nivel IES y IES×APDA.
# Unit of analysis = PROGRAMA (sin eliminar filas de merged_final).
# Merge keys: ies_norm (IES) y apda (IES×APDA).
# Outputs: Data/master_dataset.csv (utf-8-sig)
# =============================================================================

# %% Setup

import pandas as pd
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# %% Cargar archivos

print("── Cargando archivos ──")

mf = pd.read_csv('Data/merged_final.csv', encoding='utf-8-sig', low_memory=False)
print(f"merged_final:   {len(mf):,} filas × {mf.shape[1]} columnas")

anid_ies      = pd.read_csv('Data/catalogos/anid_ies.csv',      encoding='utf-8-sig')
anid_ies_apda = pd.read_csv('Data/catalogos/anid_ies_apda.csv', encoding='utf-8-sig')
sci_ies       = pd.read_csv('Data/catalogos/scimago_ies.csv',    encoding='utf-8-sig')
sci_ies_apda  = pd.read_csv('Data/catalogos/scimago_ies_apda.csv', encoding='utf-8-sig')
scival_ies_apda = pd.read_csv('Data/catalogos/scival_ies_apda.csv', encoding='utf-8-sig')
cned_programa = pd.read_csv('Data/catalogos/cned_programa.csv', encoding='utf-8-sig')
titulados_duracion = pd.read_csv('Data/catalogos/titulados_duracion.csv', encoding='utf-8-sig')

print(f"anid_ies:         {len(anid_ies)} IES")
print(f"anid_ies_apda:    {len(anid_ies_apda)} filas (IES × APDA)")
print(f"scimago_ies:      {len(sci_ies)} IES")
print(f"scimago_ies_apda: {len(sci_ies_apda)} filas (IES × APDA)")
print(f"scival_ies_apda:  {len(scival_ies_apda)} filas (IES × APDA)")
print(f"cned_programa:    {len(cned_programa)} filas (programa)")
print(f"titulados_duracion: {len(titulados_duracion)} filas (programa)")

# %% Merges — programas son las filas, no se eliminan

# ── IES×APDA: ANID ─────────────────────────────────────────────────────────
master = mf.merge(anid_ies_apda, on=['ies_norm', 'apda'], how='left')

# ── IES×APDA: Scimago ──────────────────────────────────────────────────────
master = master.merge(sci_ies_apda, on=['ies_norm', 'apda'], how='left')

# ── IES×APDA: SciVal ───────────────────────────────────────────────────────
master = master.merge(scival_ies_apda, on=['ies_norm', 'apda'], how='left')

# ── IES: ANID ──────────────────────────────────────────────────────────────
master = master.merge(anid_ies, on='ies_norm', how='left')

# ── IES: Scimago ───────────────────────────────────────────────────────────
master = master.merge(sci_ies, on='ies_norm', how='left')

# ── Programa: CNED (puntajes de selección) ─────────────────────────────────
# Match relajado a I/S/C (institución×sede×carrera), sin jornada/versión: el
# grano exacto I/S/C/J/V deja 22.3% de los programas con puntaje (muchas
# jornadas/versiones de una misma carrera no tienen su propia fila en CNED
# aunque la carrera sí participa del mismo proceso de selección). Colapsar
# CNED a I/S/C antes del merge sube la cobertura a 29.3% (+615 filas) sin
# distorsionar valores: donde ambos grados de match coinciden, acuerdan en
# 99.5% de los casos (diferencia media 0.07 puntos) — ver "Proceso de
# selección" investigation. keys_5 se mantiene para el merge de Titulados
# más abajo, que sí es sensible a jornada/versión.
keys_3 = ['cod_inst', 'cod_sede', 'cod_carrera']
keys_5 = ['cod_inst', 'cod_sede', 'cod_carrera', 'cod_jornada', 'cod_version']
cned_programa_isc = cned_programa.groupby(keys_3, as_index=False)[
    ['promedio_puntaje', 'minimo_puntaje', 'puntaje_corte']
].mean()
master = master.merge(cned_programa_isc, on=keys_3, how='left', indicator=True)
master['flag_cned'] = (master['_merge'] == 'both').astype(int)
master = master.drop(columns='_merge')

# ── Programa: Titulados (duración observada de titulación) ─────────────────
master = master.merge(titulados_duracion, on=keys_5, how='left', indicator=True)
master['flag_titulados_duracion'] = (master['_merge'] == 'both').astype(int)
master = master.drop(columns='_merge')

assert len(master) == len(mf), (
    f"ERROR: fila count cambio! merged_final={len(mf)}, master={len(master)}"
)

# %% Flags de presencia

master['flag_anid']    = master['anid_ies_n_proyectos_adjudicados'].notna().astype(int)
master['flag_scimago'] = master['scimago_ies_score_w_mean'].notna().astype(int)
master['flag_scival']  = master['scival_apda_papers'].notna().astype(int)

print(f"\n── Diagnóstico match CNED ──")
print(f"Con datos CNED (flag_cned==1): {master['flag_cned'].sum():,} "
      f"({master['flag_cned'].mean():.1%})")
print(
    master.groupby('nivel global')['flag_cned']
    .agg(['sum', 'count'])
    .assign(match_rate=lambda x: x['sum'] / x['count'])
    .rename(columns={'sum': 'con_match', 'count': 'total'})
    .to_string()
)

print(f"\n── Diagnóstico match Titulados (duración) ──")
print(f"Con datos titulados (flag_titulados_duracion==1): "
      f"{master['flag_titulados_duracion'].sum():,} "
      f"({master['flag_titulados_duracion'].mean():.1%})")
print(
    master.groupby('nivel global')['flag_titulados_duracion']
    .agg(['sum', 'count'])
    .assign(match_rate=lambda x: x['sum'] / x['count'])
    .rename(columns={'sum': 'con_match', 'count': 'total'})
    .to_string()
)

# %% Matrícula share por APDA (nacional, sin umbral) y proporción posgrado IES

_apda_valid = master[master['apda'] != 'No aplica']
_total_por_apda = _apda_valid.groupby('apda')['total matrícula'].sum()
_total_sistema = _total_por_apda.sum()
share_apda = (
    (_total_por_apda / _total_sistema * 100)
    .rename('matricula_share_apda')
    .reset_index()
)
master = master.merge(share_apda, on='apda', how='left')

# prop_matricula_posgrado_ies — retirada 2026-08-18: institution-constant,
# sin variación real intra-institución. Reemplazada por
# prop_matricula_posgrado_apda/prop_matricula_doctorado_apda (razón real por
# celda IES×APDA, ver Utils/data_prep.py::add_prop_matricula_posgrado_apda).
# Dejada calculada aquí por continuidad, ya no se usa en config.yaml.
_total_matricula_ies = master.groupby('ies_norm')['total matrícula'].sum()
_posgrado_matricula_ies = (
    master[master['nivel global'] == 'Posgrado']
    .groupby('ies_norm')['total matrícula'].sum()
)
prop_posgrado_ies = (
    (_posgrado_matricula_ies / _total_matricula_ies)
    .reindex(_total_matricula_ies.index)
    .fillna(0.0)
    .rename('prop_matricula_posgrado_ies')
    .reset_index()
)
master = master.merge(prop_posgrado_ies, on='ies_norm', how='left')

# %% Diagnósticos de match rate

n_total = len(master)
print(f"\n── Diagnósticos de match ──")
print(f"Total filas (programas): {n_total:,}")
print(f"  Con datos ANID    (flag_anid==1):    {master['flag_anid'].sum():,}  "
      f"({master['flag_anid'].mean():.1%})")
print(f"  Con datos Scimago (flag_scimago==1): {master['flag_scimago'].sum():,}  "
      f"({master['flag_scimago'].mean():.1%})")
print(f"  Con datos SciVal  (flag_scival==1):  {master['flag_scival'].sum():,}  "
      f"({master['flag_scival'].mean():.1%})")
print(f"  Con ambos (ANID y Scimago):           "
      f"{(master['flag_anid'] & master['flag_scimago']).sum():,}")

# IES sin match ANID
print("\n── IES sin datos ANID ──")
sin_anid = (
    master[master['flag_anid'] == 0]['ies_norm']
    .dropna().unique()
)
for n in sorted(sin_anid):
    print(f"  {n}")

# IES sin match Scimago
print("\n── IES sin datos Scimago ──")
sin_sci = (
    master[master['flag_scimago'] == 0]['ies_norm']
    .dropna().unique()
)
for n in sorted(sin_sci):
    print(f"  {n}")

# %% Columnas nuevas añadidas

new_cols = [c for c in master.columns if c not in mf.columns]
print(f"\n── Columnas nuevas añadidas ({len(new_cols)}) ──")
for c in new_cols:
    pct_nn = master[c].notna().mean() * 100 if master[c].dtype != object else None
    if pct_nn is not None:
        print(f"  {c:<45} {pct_nn:.1f}% non-null")
    else:
        print(f"  {c}")

# %% UFT sanity check

uft = master[master['ies_norm'].str.contains('finis terrae', na=False)]
print(f"\n── UFT sanity check ──")
print(f"  Filas UFT: {len(uft)}")
print(f"  flag_anid:    {uft['flag_anid'].sum()} / {len(uft)}")
print(f"  flag_scimago: {uft['flag_scimago'].sum()} / {len(uft)}")
if len(uft) > 0:
    uft_row = uft.iloc[0]
    print(f"  anid_ies_n_proyectos_adjudicados: {uft_row.get('anid_ies_n_proyectos_adjudicados', 'n/a')}")
    print(f"  scimago_ies_score_w_mean: {uft_row.get('scimago_ies_score_w_mean', 'n/a')}")

# %% Exportar

os.makedirs('dfs', exist_ok=True)
master.to_csv('Data/master_dataset.csv', index=False, encoding='utf-8-sig')
print(f"\nExportado: Data/master_dataset.csv  ({len(master):,} filas × {master.shape[1]} columnas)")
