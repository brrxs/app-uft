# =============================================================================
# Exp_CNED.py
# Construye el catálogo CNED a nivel programa (I/S/C/J/V) con puntajes de
# selección: promedio_puntaje, minimo_puntaje, puntaje_corte.
# Output: Data/catalogos/cned_programa.csv
# =============================================================================

# %% Setup

import pandas as pd
import sys
import os
import re

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# %% Load

periodo = 2025

cned = pd.read_excel('datos/CNED/BaseINDICES-2005-2025.xlsx')
print(f"CNED crudo: {cned.shape[0]:,} filas × {cned.shape[1]} columnas")

# %% Filtrar universidades, año vigente

cned_u = cned[
    (cned['Tipo Institución'] == 'Univ.')
    & (cned['Año'] == periodo)
].copy()

print(f"CNED universidades {periodo}: {cned_u.shape[0]:,} filas")

# %% Descomponer Códgo SIES en las 5 keys (mismo patrón que Exp_SIES.py)

keys_5 = ['cod_inst', 'cod_sede', 'cod_carrera', 'cod_jornada', 'cod_version']
pattern = r'I(\d+)S(\d+)C(\d+)J(\d+)V(\d+)'

keys_parsed = cned_u['Códgo SIES'].str.extract(pattern)
keys_parsed.columns = keys_5
keys_parsed = keys_parsed.astype('Int64')

cned_u = pd.concat([cned_u.reset_index(drop=True), keys_parsed.reset_index(drop=True)], axis=1)

n_sin_key = cned_u[keys_5].isna().any(axis=1).sum()
print(f"Filas sin key I/S/C/J/V parseable: {n_sin_key}")

# %% Seleccionar y renombrar columnas de puntaje

cols_puntaje = {
    'Promedio Puntaje (promedio matemáticas y lenguaje)': 'promedio_puntaje',
    'Mínimo Puntaje (promedio matemáticas y lenguaje)':   'minimo_puntaje',
    'Puntaje de corte (promedio de la carrera)':          'puntaje_corte',
}

cned_slim = cned_u[keys_5 + list(cols_puntaje.keys())].rename(columns=cols_puntaje)
cned_slim = cned_slim.dropna(subset=keys_5).drop_duplicates(subset=keys_5)

print(f"\ncned_programa: {cned_slim.shape[0]:,} filas (1 por I/S/C/J/V)")
for c in cols_puntaje.values():
    print(f"  {c:<20} {cned_slim[c].notna().mean():.1%} cobertura")

# %% Exportar

os.makedirs('Data/catalogos', exist_ok=True)
cned_slim.to_csv('Data/catalogos/cned_programa.csv', index=False, encoding='utf-8-sig')
print("\nExportado: Data/catalogos/cned_programa.csv")
