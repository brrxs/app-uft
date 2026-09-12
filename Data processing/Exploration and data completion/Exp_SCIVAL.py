# %% Setup

import pandas as pd
import sys
import os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

exec(open('Utils/utils.py', encoding='utf-8').read())

# %% Load

scival = pd.read_csv('datos/SciVal/scival_subject_corrected_2025.csv', encoding='utf-8-sig')

print(f"scival: {scival.shape[0]:,} filas × {scival.shape[1]} columnas")
print(f"{scival['universidad'].nunique()} universidades, {scival['subject'].nunique()} subjects")
print(scival['apda'].value_counts(dropna=False))

# %% Cleaning — apda spelling + drop "No aplica"

# La fuente escribe "tecnologia" sin tilde; master_dataset usa "tecnología".
scival['apda'] = scival['apda'].replace({
    'Ciencias, tecnologia y calidad de vida': 'Ciencias, tecnología y calidad de vida',
})

scival_apda = scival[scival['apda'] != 'No aplica'].copy()

# %% universidad → ies_norm

# Diccionario de correcciones para nombres que no coinciden después de normalizar
# (mismo patrón que _scimago_fix en Exp_SCIMAGO.py)
_scival_fix = {
    'universidad catolica silva henriquez':        'universidad catolica cardenal raul silva henriquez',
    'universidad internacional sek chile':          'universidad sek',
    'universidad la republica, chile':              'universidad la republica',
    'universidad miguel de cervantes, chile':       'universidad miguel de cervantes',
    'universidad santo tomas, santiago':             'universidad santo tomas',
    'universidad tecnologica de chile':             'universidad tecnologica de chile inacap',
    'universidad vina del mar':                     'universidad de vina del mar',
    'universidad de artes, ciencias y comunicacion': 'universidad de artes, ciencias y comunicacion uniacc',
    'universidad de las americas - chile':          'universidad de las americas',
    'universidad de los andes chile':               'universidad de los andes',
}


def _ies_norm_scival(nombre):
    raw = normalizar_nombre_ies(nombre)
    return _scival_fix.get(raw, raw)


scival_apda['ies_norm'] = scival_apda['universidad'].apply(_ies_norm_scival)

# Diagnóstico: instituciones sin match en merged_final (esperado: universidades
# fuera del universo activo, ej. cerradas o no acreditadas)
mf_norms = set(pd.read_csv('Data/merged_final.csv', encoding='utf-8-sig',
                            low_memory=False)['ies_norm'].unique())
_unmatched_scival = sorted({
    u for u in scival_apda['universidad'].unique()
    if _ies_norm_scival(u) not in mf_norms
})
print("\n── SciVal instituciones sin match en merged_final ──")
if _unmatched_scival:
    for u in _unmatched_scival:
        print(f"  {u!r}  →  ies_norm: {_ies_norm_scival(u)!r}")
else:
    print("  (ninguna)")

# %% Agregar a nivel IES × APDA — suma de papers a través de los subjects de cada APDA

scival_ies_apda = (
    scival_apda
    .groupby(['ies_norm', 'apda'], as_index=False)
    .agg(scival_apda_papers=('valor_corregido_2025', 'sum'))
)

print(f"\nscival_ies_apda: {scival_ies_apda.shape[0]} filas  "
      f"({scival_ies_apda['ies_norm'].nunique()} IES × {scival_ies_apda['apda'].nunique()} APDAs)")

# %% Exportar

os.makedirs('Data/catalogos', exist_ok=True)
scival_ies_apda.to_csv('Data/catalogos/scival_ies_apda.csv', index=False, encoding='utf-8-sig')
print("Exportado: Data/catalogos/scival_ies_apda.csv")
