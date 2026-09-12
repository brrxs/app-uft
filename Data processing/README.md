# APP UFT — procesamiento de datos APDA

Copia autonoma del pipeline de preparacion de datos de `Proyecto APDAs` (rama
`refactor/core-components`). A diferencia de `../Dashboard/` (que solo lee
scores, memberships y perfiles ya calculados), esta carpeta reconstruye esas
tablas de punta a punta: desde los archivos crudos en `datos/` hasta
`Data/master_dataset.csv`, y de ahi hasta scores y arquetipos. No incluye los
scripts de diagnostico (`04_no_imputation.py`, `06_stability_check.py`,
`07_archetypoids.py`, `Indicators_register.py`, `Exp_PAC_JCE_check.py`) —
fuera de alcance; ver `Proyecto APDAs` si se necesitan.

## Setup

Requiere Python 3.14 (mismo interprete que `Dashboard/`, el unico con el
paquete `archetypes` instalado en la maquina original:
`C:\Users\clien\AppData\Local\Programs\Python\Python314\python.exe`).

```
pip install -r requirements.txt
```

## Generar las tablas

Correr siempre desde la raiz de esta carpeta (`APP UFT/Data processing`).

### 1. Dato crudo -> master_dataset.csv

Orden obligatorio (algunos scripts leen la salida de otro):

```
python "Exploration and data completion/Exp_Titulados_Generica.py"
python "Exploration and data completion/Exp_SIES.py"

# Estos tres son independientes entre si y de los dos anteriores:
python "Exploration and data completion/Exp_ANID.py"
python "Exploration and data completion/Exp_CNED.py"
python "Exploration and data completion/Exp_Titulados.py"

# Necesitan Data/merged_final.csv (lo escribe Exp_SIES.py):
python "Exploration and data completion/Exp_SCIMAGO.py"
python "Exploration and data completion/Exp_SCIVAL.py"

# Al final, combina todo en Data/master_dataset.csv:
python "Exploration and data completion/Build_master_dataset.py"
```

Estos scripts abren figuras de diagnostico con matplotlib; en esta copia usan
el backend no interactivo `Agg`, asi que corren sin ventanas emergentes.

### 2. master_dataset.csv -> features, scores y arquetipos

```
python prepare_data.py vector
python prepare_data.py archetypes
python prepare_data.py cine
# o bien, las tres de una vez:
python prepare_data.py all

python Vector/01_vectors.py

python Archetypes/01_dimension_scores.py
python Archetypes/01_dimension_scores.py Cine/config.yaml Cine/outputs

python Archetypes/02_archetypes.py
python Archetypes/02_archetypes.py Cine/config.yaml Cine/outputs
```

`Cine/` no tiene sus propios scripts de scores/arquetipos: reusa los de
`Archetypes/` pasandoles su propio `config.yaml` y carpeta de salida, igual
que en `Proyecto APDAs`.

## Publicar los resultados en Dashboard/

`../Dashboard/` es una copia independiente, no lee de aca en vivo. Despues de
regenerar, copiar a mano:

| Desde (`Data processing/`) | Hacia (`Dashboard/`) |
|---|---|
| `Vector/data/ies_apda_features.csv` | `Vector/data/ies_apda_features.csv` |
| `Vector/outputs/dimension_scores.csv` | `Vector/outputs/dimension_scores.csv` |
| `Archetypes/outputs/dimension_scores.csv` | `Archetypes/outputs/dimension_scores.csv` |
| `Archetypes/outputs/archetype_memberships.csv` | `Archetypes/outputs/archetype_memberships.csv` |
| `Archetypes/outputs/archetype_profiles_*.csv` (4) | `Archetypes/outputs/` |
| `Cine/data/ies_cine_features.csv` | `Cine/data/ies_cine_features.csv` |
| `Cine/outputs/dimension_scores.csv` | `Cine/outputs/dimension_scores.csv` |
| `Cine/outputs/archetype_memberships.csv` | `Cine/outputs/archetype_memberships.csv` |
| `Cine/outputs/archetype_profiles_*.csv` (9) | `Cine/outputs/` |

Luego generar los dashboards como documenta `Dashboard/README.md`.

## Contenido

| Carpeta / archivo | Contiene |
|---|---|
| `datos/` | Archivos crudos (~1.3 GB): `ANID/`, `CNED/`, `MINEDUC/`, `Retención/`, `SIES/`, `SciVal/`, `Scimago/` |
| `Exploration and data completion/` | Scripts que arman `Data/merged_final.csv`, `Data/catalogos/*` y `Data/master_dataset.csv` desde `datos/` |
| `Data/` | `master_dataset.csv` (insumo final) + `merged_final.csv` y `catalogos/` (intermedios) |
| `Utils/` | `data_prep.py`, `prepare_features.py`, `utils.py`, `archetypes_compat.py` |
| `Vector/` | Config + script de la rama principal |
| `Archetypes/` | Config + scripts de dimension scores / arquetipos (reusados por Cine) |
| `Cine/` | Config de la rama por area CINE-F 13 |
| `prepare_data.py` | Reemplaza los tres `00_prepare_data.py` del repo original (uno por rama) |

## Actualizar esta copia

`Proyecto APDAs` no se toca por este trabajo: todos sus scripts originales
(`Exp_*.py`, `Build_master_dataset.py`, los `00_prepare_data.py`) siguen
igual. Si los archivos en `datos/` cambian alla, copiarlos de nuevo aca (no
estan en git: `datos/` esta en `.gitignore` del repo original) y volver a
correr los pasos de esta pagina.
