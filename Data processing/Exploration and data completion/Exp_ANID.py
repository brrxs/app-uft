# %% Setup

import numpy as np
import pandas as pd
import sys
import os
import re

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

exec(open('Utils/utils.py', encoding='utf-8').read())

# %% Load

anid = pd.read_excel('datos/ANID/BDH_HISTORICA.xlsx')

# %% Inspect columns

print("═" * 60)
print(f"SHAPE: {anid.shape[0]:,} filas × {anid.shape[1]} columnas")
print("═" * 60)

col_info = pd.DataFrame({
    'dtype':         [str(anid[c].dtype) for c in anid.columns],
    'n_no_null':     [int(anid[c].notna().sum()) for c in anid.columns],
    'cobertura_pct': [round(anid[c].notna().mean() * 100, 1) for c in anid.columns],
    'n_unique':      [anid[c].nunique() for c in anid.columns],
})
col_info.index = anid.columns

print(col_info.to_string())

# %% Normalize institutions — all rows

anid['ies_norm'] = anid['INSTITUCION_PRINCIPAL'].apply(normalizar_nombre_ies)
anid['ies_abrev'] = anid['ies_norm'].map(abreviaciones_ies)  # NaN if unmapped

# %% Duración imputada y ventana de actividad 2025
#
# BDH_HISTORICA no trae fechas exactas de inicio/fin, solo AGNO_FALLO (año de
# adjudicación) y DURACION_MESES (float, ~4.8% nulo). Para saber si un
# proyecto sigue "activo" en 2025 (más allá de haber sido adjudicado ese año)
# aproximamos a nivel de año: se asume que el proyecto parte en AGNO_FALLO y
# dura ceil(DURACION_MESES/12) años. Donde falta DURACION_MESES, se imputa
# con la mediana del mismo PROGRAMA+INSTRUMENTO (o, si esa combinación no
# tiene ningún valor no-nulo, con la mediana a nivel de PROGRAMA) — calculado
# sobre el dataset completo (todas las instituciones) para maximizar el
# tamaño muestral de la imputación.

anid['dur_imputada'] = anid['DURACION_MESES'].fillna(
    anid.groupby(['PROGRAMA', 'INSTRUMENTO'])['DURACION_MESES'].transform('median')
)
anid['dur_imputada'] = anid['dur_imputada'].fillna(
    anid.groupby('PROGRAMA')['DURACION_MESES'].transform('median')
)

anid['agno_fin'] = np.maximum(
    anid['AGNO_FALLO'],
    anid['AGNO_FALLO'] + np.ceil(anid['dur_imputada'] / 12) - 1,
)
anid['activo_2025'] = (anid['AGNO_FALLO'] <= 2025) & (anid['agno_fin'] >= 2025)

# Summary: universities only (todas las agnos — el filtro por año se aplica
# más abajo, por separado, para las vistas "adjudicados" y "activos")
_pat_univ = re.compile(r'universidad')

anid_univ = anid[anid['ies_norm'].str.contains(_pat_univ, na=False)]

# Vista "adjudicados 2025" — usada para las secciones exploratorias de abajo
# (mismo comportamiento que antes de agregar la ventana de actividad).
anid_univ_2025 = anid_univ[anid_univ['AGNO_FALLO'] == 2025]

ies_summary = (
    anid_univ_2025.groupby('ies_norm')
    .agg(
        n_proyectos=('ies_norm', 'count'),
        ies_abrev=('ies_abrev', 'first'),
    )
    .sort_values('n_proyectos', ascending=False)
    .reset_index()
)
ies_summary['mapeada'] = ies_summary['ies_abrev'].notna()

print("\n\n═" * 30)
print("INSTITUCIONES EN EL DATASET")
print("═" * 60)
print(ies_summary.to_string(index=False))

# Unmapped institutions
no_map = ies_summary[~ies_summary['mapeada']]
print(f"\n── Sin mapeo en abreviaciones_ies: {len(no_map)} instituciones ──")
print(no_map[['ies_norm', 'n_proyectos']].to_string(index=False))



# %% APDAs

print("\n".join(anid_univ_2025.columns.tolist()))
tab(anid_univ_2025['AREA_OCDE'])


ocde_to_apda = {
    'CIENCIAS MEDICAS Y DE LA SALUD': 'Salud y bienestar',
    'HUMANIDADES':                     'Artes, humanidades y cultura',
    'CIENCIAS SOCIALES':               'Ed., Sociedad y desarrollo humano',
    'CIENCIAS NATURALES':              'Ciencias, tecnología y calidad de vida',
    'INGENIERIA Y TECNOLOGIA':         'Ciencias, tecnología y calidad de vida',
    'CIENCIAS AGRICOLAS':              'No aplica',
    'MULTIDISCIPLINARIO':              'No aplica',
    'NO APLICA':                       'No aplica',
    'SIN INFORMACION':                 'No aplica',
}

anid_univ['APDA'] = anid_univ['AREA_OCDE'].map(ocde_to_apda).fillna('No aplica')
tab(anid_univ['APDA'])


# %% Column catalogue

col_cat = pd.DataFrame({
    'dtype':     [str(anid_univ[c].dtype) for c in anid_univ.columns],
    'nan_pct':   [round(anid_univ[c].isna().mean() * 100, 1) for c in anid_univ.columns],
    'n_unique':  [anid_univ[c].nunique() for c in anid_univ.columns],
})
col_cat.index = anid_univ.columns

print("═" * 60)
print(" "*20,"CATÁLOGO DE COLUMNAS")
print("═" * 60)
print(col_cat.to_string())
col_cat.to_excel('Data/anid_col_catalogue.xlsx')


# %% Agregar a nivel IES y IES×APDA

# Renombrar la columna APDA a apda (minúsculas) para consistencia con merged_final
anid_univ = anid_univ.copy()
anid_univ['apda'] = anid_univ['APDA']

# Categoría de fondo: fondecyt (Regular) / otros_fondecyt (otros instrumentos
# dentro de FONDECYT, ej. Iniciación, Postdoctorado, Doctorado) / no_fondecyt
anid_univ['fondecyt_cat'] = np.select(
    [
        (anid_univ['PROGRAMA'] == 'FONDECYT') & (anid_univ['INSTRUMENTO'] == 'REGULAR'),
        (anid_univ['PROGRAMA'] == 'FONDECYT') & (anid_univ['INSTRUMENTO'] != 'REGULAR'),
    ],
    ['fondecyt', 'otros_fondecyt'],
    default='no_fondecyt',
)
tab(anid_univ['fondecyt_cat'])

# ── Desglose por categoría fondecyt / no_fondecyt / otros_fondecyt ─────────

def _cat_pivot(df_src, group_cols, prefix, suffix):
    """n_proyectos y monto_mean por fondecyt_cat, pivotados a columnas anchas."""
    piv = (
        df_src.groupby(group_cols + ['fondecyt_cat'])
        .agg(
            n_proyectos=('CODIGO_PROYECTO', 'count'),
            monto_mean=('MONTO_ADJUDICADO', 'mean'),
        )
        .unstack('fondecyt_cat')
    )
    piv.columns = [f'{prefix}_{stat}_{cat}_{suffix}' for stat, cat in piv.columns]
    n_cols = [c for c in piv.columns if '_n_proyectos_' in c]
    piv[n_cols] = piv[n_cols].fillna(0)
    return piv.reset_index()


def _aggregate_view(df_view, suffix):
    """Agrega df_view (ya filtrado a la ventana temporal deseada) a nivel IES
    y IES×APDA, con todas las columnas producidas sufijadas por `suffix`
    (p.ej. 'adjudicados' o 'activos')."""
    ies = (
        df_view.groupby('ies_norm')
        .agg(**{
            f'anid_ies_n_proyectos_{suffix}':   ('CODIGO_PROYECTO',  'count'),
            f'anid_ies_monto_sum_{suffix}':     ('MONTO_ADJUDICADO', 'sum'),
            f'anid_ies_monto_mean_{suffix}':    ('MONTO_ADJUDICADO', 'mean'),
            f'anid_ies_duracion_mean_{suffix}': ('DURACION_MESES',   'mean'),
        })
        .reset_index()
    )
    ies_apda = (
        df_view.groupby(['ies_norm', 'apda'])
        .agg(**{
            f'anid_apda_n_proyectos_{suffix}':   ('CODIGO_PROYECTO',  'count'),
            f'anid_apda_monto_sum_{suffix}':     ('MONTO_ADJUDICADO', 'sum'),
            f'anid_apda_monto_mean_{suffix}':    ('MONTO_ADJUDICADO', 'mean'),
            f'anid_apda_duracion_mean_{suffix}': ('DURACION_MESES',   'mean'),
        })
        .reset_index()
    )

    ies_cat = _cat_pivot(df_view, ['ies_norm'], 'anid_ies', suffix)
    ies_apda_cat = _cat_pivot(df_view, ['ies_norm', 'apda'], 'anid_apda', suffix)

    ies = ies.merge(ies_cat, on='ies_norm', how='left')
    ies_apda = ies_apda.merge(ies_apda_cat, on=['ies_norm', 'apda'], how='left')
    return ies, ies_apda


# Vista "adjudicados": AGNO_FALLO == 2025 (comportamiento histórico)
anid_univ_adjudicados = anid_univ[anid_univ['AGNO_FALLO'] == 2025]
# Vista "activos": proyectos cuya ventana [AGNO_FALLO, agno_fin] cubre 2025,
# incluye proyectos plurianuales adjudicados en años anteriores aún vigentes
anid_univ_activos = anid_univ[anid_univ['activo_2025']]

anid_ies_adj, anid_ies_apda_adj = _aggregate_view(anid_univ_adjudicados, 'adjudicados')
anid_ies_act, anid_ies_apda_act = _aggregate_view(anid_univ_activos, 'activos')

anid_ies = anid_ies_adj.merge(anid_ies_act, on='ies_norm', how='outer')
anid_ies_apda = anid_ies_apda_adj.merge(anid_ies_apda_act, on=['ies_norm', 'apda'], how='outer')

print(f"\nanid_ies:      {anid_ies.shape[0]} IES")
print(f"anid_ies_apda: {anid_ies_apda.shape[0]} filas (IES × APDA)")
print(anid_ies.head())
print(anid_ies_apda.head())

# ── Exportar ────────────────────────────────────────────────────────────────
import os
os.makedirs('Data/catalogos', exist_ok=True)

anid_ies.to_csv('Data/catalogos/anid_ies.csv',           index=False, encoding='utf-8-sig')
anid_ies_apda.to_csv('Data/catalogos/anid_ies_apda.csv', index=False, encoding='utf-8-sig')

print("Exportado: Data/catalogos/anid_ies.csv")
print("Exportado: Data/catalogos/anid_ies_apda.csv")
