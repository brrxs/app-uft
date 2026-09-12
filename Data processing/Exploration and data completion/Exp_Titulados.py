# =============================================================================
# Exp_Titulados.py
# Construye el catálogo de duración observada de titulación a nivel programa
# (I/S/C/J/V), a partir de Titulados Ed. Superior MINEDUC 2024.
# Duración = fecha_obtencion_titulo - fecha de ingreso (marzo/agosto según
# sem_ing_carr_ori), sólo para titulados con anio/sem de ingreso ori == act
# (sin cambios de carrera/readmisión que distorsionen el ingreso original).
# Output: Data/catalogos/titulados_duracion.csv
# =============================================================================

# %% Setup

import pandas as pd
import numpy as np
import sys
import os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# %% Load

titulados = pd.read_csv(
    'datos/MINEDUC/20250718_Titulados_Ed_Superior_2024_WEB.csv',
    sep=';',
    encoding='utf-8',
)
print(f"Titulados crudo: {titulados.shape[0]:,} filas × {titulados.shape[1]} columnas")

# %% Filtrar registros válidos
# mrun != 0, sólo universidades, anio_ing_carr_ori conocido (!= 1900) y
# consistente con anio_ing_carr_act, sem_ing_carr_ori conocido (!= 0) y
# consistente con sem_ing_carr_act.

mask = (
    (titulados['mrun'] != 0)
    & (titulados['tipo_inst_1'] == 'Universidades')
    & (titulados['anio_ing_carr_ori'] != 1900)
    & (titulados['anio_ing_carr_ori'] == titulados['anio_ing_carr_act'])
    & (titulados['sem_ing_carr_ori'] != 0)
    & (titulados['sem_ing_carr_ori'] == titulados['sem_ing_carr_act'])
)
tit_f = titulados[mask].copy()

print(f"  mrun != 0:                      {(titulados['mrun'] != 0).sum():,}")
print(f"  tipo_inst_1 == Universidades:   {(titulados['tipo_inst_1'] == 'Universidades').sum():,}")
print(f"  anio_ing_carr_ori != 1900:      {(titulados['anio_ing_carr_ori'] != 1900).sum():,}")
print(f"  sem_ing_carr_ori != 0:          {(titulados['sem_ing_carr_ori'] != 0).sum():,}")
print(f"Tras todos los filtros: {tit_f.shape[0]:,} filas "
      f"({tit_f.shape[0] / titulados.shape[0]:.1%})")

# %% Calcular fecha de inicio (convención marzo/agosto) y fecha de titulación

tit_f['mes_ini'] = np.where(tit_f['sem_ing_carr_ori'] == 1, 3, 8)
tit_f['start_date'] = pd.to_datetime(
    dict(year=tit_f['anio_ing_carr_ori'], month=tit_f['mes_ini'], day=1)
)
tit_f['end_date'] = pd.to_datetime(
    tit_f['fecha_obtencion_titulo'].astype(int).astype(str),
    format='%Y%m%d',
    errors='coerce',
)

n_sin_fecha = tit_f['end_date'].isna().sum()
print(f"\nfecha_obtencion_titulo no parseable: {n_sin_fecha}")
tit_f = tit_f.dropna(subset=['end_date'])

tit_f['duration_years'] = (tit_f['end_date'] - tit_f['start_date']).dt.days / 365.25

# %% Sanity floor: descartar duraciones <= 0

n_before = len(tit_f)
tit_f = tit_f[tit_f['duration_years'] > 0]
n_dropped = n_before - len(tit_f)
print(f"Duraciones <= 0 descartadas: {n_dropped} ({n_dropped / n_before:.2%})")

print("\nDistribución duration_years:")
print(tit_f['duration_years'].describe())
print(tit_f['duration_years'].quantile([.01, .05, .5, .95, .99]))

# %% Diagnóstico nivel_global (informativo, sin filtrar por él: codigo_unico
# ya es específico de un programa/nivel, así que la agregación por
# codigo_unico mantiene Pregrado/Posgrado/Postítulo separados)

print("\nnivel_global (1 fila por codigo_unico):")
print(tit_f.drop_duplicates('codigo_unico')['nivel_global'].value_counts())

# %% Agregar a nivel codigo_unico

titulados_dur = (
    tit_f
    .groupby('codigo_unico', as_index=False)
    .agg(
        duracion_titulados_mean=('duration_years', 'mean'),
        duracion_titulados_median=('duration_years', 'median'),
        duracion_titulados_std=('duration_years', 'std'),
        duracion_titulados_n=('duration_years', 'size'),
    )
)
print(f"\ntitulados_duracion (antes de decodificar keys): {titulados_dur.shape[0]:,} programas (codigo_unico)")

# %% Descomponer codigo_unico en las 5 keys (mismo patrón que Exp_SIES.py / Exp_CNED.py)

keys_5 = ['cod_inst', 'cod_sede', 'cod_carrera', 'cod_jornada', 'cod_version']
pattern = r'I(\d+)S(\d+)C(\d+)J(\d+)V(\d+)'

keys_parsed = titulados_dur['codigo_unico'].str.extract(pattern)
keys_parsed.columns = keys_5
keys_parsed = keys_parsed.astype('Int64')

titulados_dur = pd.concat(
    [titulados_dur.reset_index(drop=True), keys_parsed.reset_index(drop=True)], axis=1
)

n_sin_key = titulados_dur[keys_5].isna().any(axis=1).sum()
print(f"Filas sin key I/S/C/J/V parseable: {n_sin_key}")

titulados_dur = titulados_dur.dropna(subset=keys_5).drop_duplicates(subset=keys_5)
print(f"titulados_duracion final: {titulados_dur.shape[0]:,} filas (1 por I/S/C/J/V)")

# %% Exportar

cols_out = keys_5 + [
    'duracion_titulados_mean', 'duracion_titulados_median',
    'duracion_titulados_std', 'duracion_titulados_n',
]
titulados_dur = titulados_dur[cols_out]

os.makedirs('Data/catalogos', exist_ok=True)
titulados_dur.to_csv('Data/catalogos/titulados_duracion.csv', index=False, encoding='utf-8-sig')
print("\nExportado: Data/catalogos/titulados_duracion.csv")
