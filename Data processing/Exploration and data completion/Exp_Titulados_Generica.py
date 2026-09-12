# =============================================================================
# Exp_Titulados_Generica.py
# Construye el catálogo de titulados por (institución × carrera genérica),
# en dos ventanas de 3 años, para ponderar empleabilidad_1er_año/2do_año por
# volumen de egresados en vez de por matrícula (ver Utils/data_prep.py::
# add_empleabilidad_ponderada). Los años se infieren de los nombres de los
# archivos MINEDUC — no están hardcodeados.
#
# titulados_gen_emp1 = suma de los 3 años más recientes (pesa empleabilidad
#                       1er año, que mide la cohorte titulada esos mismos años)
# titulados_gen_emp2 = suma de los 3 años siguientes, un año más atrás (pesa
#                       empleabilidad 2do año, que mide un año después de
#                       titulación)
#
# La genérica de cada codigo_unico se toma del mapa CÓDIGO CARRERA → ÁREA
# CARRERA GENÉRICA de SIES Matrícula (misma taxonomía que 'Nombre carrera
# genérica' del Buscador de Empleabilidad y que 'área carrera genérica' en
# Data/master_dataset.csv — verificado 100% de acuerdo en las filas con
# empleabilidad). Se usa el mapa en vez de area_generica propio de Titulados
# porque no comparte vocabulario con el Buscador de Empleabilidad.
#
# Output: Data/catalogos/titulados_generica.csv
# =============================================================================

# %% Setup

import glob
import os
import re
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# %% Descubrir archivos-año de Titulados MINEDUC

_paths = glob.glob('datos/MINEDUC/*Titulados_Ed_Superior_*_WEB.csv')
_year_pattern = re.compile(r'_(\d{4})_WEB\.csv$')
years_paths = sorted(
    ((int(_year_pattern.search(p).group(1)), p) for p in _paths),
    reverse=True,
)
years = [y for y, _ in years_paths]
print(f"Archivos Titulados encontrados: {years_paths}")

W1 = years[:3]   # pesa empleabilidad_1er_año
W2 = years[1:4]  # pesa empleabilidad_2do_año
print(f"Ventana empleabilidad_1er_año (W1): {W1}")
print(f"Ventana empleabilidad_2do_año (W2): {W2}")

# %% Mapa código carrera → carrera genérica (SIES Matrícula, etiqueta más reciente)

mat = pd.read_csv(
    'datos/SIES/Matricula_2007_2025_WEB_15_07_2025.csv',
    sep=';',
    encoding='latin-1',
    usecols=['AÑO', 'CÓDIGO CARRERA', 'ÁREA CARRERA GENÉRICA'],
    low_memory=False,
)
mat['_year'] = mat['AÑO'].str[-4:].astype(int)
mat = mat.sort_values('_year')
generica_map = (
    mat.drop_duplicates('CÓDIGO CARRERA', keep='last')
       .set_index('CÓDIGO CARRERA')['ÁREA CARRERA GENÉRICA']
)
print(f"Mapa código carrera → genérica: {len(generica_map):,} códigos")

# %% Cargar y agregar cada archivo-año (sólo Pregrado, único ámbito de empleabilidad)

counts_por_año = {}
for year, path in years_paths:
    tit = pd.read_csv(
        path,
        sep=';',
        encoding='utf-8',
        usecols=['cat_periodo', 'codigo_unico', 'cod_inst', 'nivel_global'],
        low_memory=False,
    )
    tit = tit[tit['nivel_global'] == 'Pregrado'].copy()
    tit['carrera_generica'] = tit['codigo_unico'].map(generica_map)

    n_sin_map = tit['carrera_generica'].isna().sum()
    print(f"  {year}: {len(tit):,} titulados Pregrado, "
          f"sin mapa de genérica: {n_sin_map:,} ({n_sin_map / len(tit):.2%})")
    tit = tit.dropna(subset=['carrera_generica'])

    counts_por_año[year] = tit.groupby(['cod_inst', 'carrera_generica']).size()

# %% Pivotar a ancho y sumar las dos ventanas

wide = pd.DataFrame(counts_por_año).fillna(0)
wide.columns = [f'tit_{y}' for y in wide.columns]

titulados_gen = wide.copy()
titulados_gen['titulados_gen_emp1'] = wide[[f'tit_{y}' for y in W1]].sum(axis=1)
titulados_gen['titulados_gen_emp2'] = wide[[f'tit_{y}' for y in W2]].sum(axis=1)
titulados_gen = titulados_gen.reset_index()

print(f"\ntitulados_generica: {len(titulados_gen):,} pares (cod_inst, carrera_generica)")
print(titulados_gen[['titulados_gen_emp1', 'titulados_gen_emp2']].describe())

# %% Exportar

cols_out = ['cod_inst', 'carrera_generica', 'titulados_gen_emp1', 'titulados_gen_emp2']
titulados_gen = titulados_gen[cols_out]

os.makedirs('Data/catalogos', exist_ok=True)
titulados_gen.to_csv('Data/catalogos/titulados_generica.csv', index=False, encoding='utf-8-sig')
print("\nExportado: Data/catalogos/titulados_generica.csv")
