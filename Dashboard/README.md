# APP UFT — dashboards APDA

Copia autonoma del generador de dashboards de `Proyecto APDAs` (rama
`refactor/core-components`, copiada 2026-09-11). Trae solo lo que
`Dashboard/build_dashboard.py` necesita para leer: scores, memberships y
perfiles de arquetipos ya calculados. No reconstruye nada desde datos crudos
ni vuelve a correr los pasos `00_prepare_data.py` / `01_*` / `02_*` del
proyecto original.

## Setup

Requiere Python 3.14 (el unico interprete con el paquete `archetypes`
instalado en la maquina original: `C:\Users\clien\AppData\Local\Programs\Python\Python314\python.exe`).

```
pip install -r requirements.txt
```

## Generar los dashboards

Correr siempre desde la raiz de esta carpeta (`APP UFT/Dashboard`):

```
python build_dashboard.py                    # dashboard principal -> outputs/index.html
                                               #   + BI testing/ (paginas para Power BI)
                                               #   + site/apda/*.html (paginas del sitio unificado)
python build_dashboard.py Cine/config.yaml    # dashboard por area CINE -> Cine/outputs/index.html
                                               #   + site/cine/*.html
python build_site.py                          # site/index.html (menu) + chequeo de links
```

Los tres comandos, en ese orden, arman `site/`: un sitio estatico navegable con
un menu principal, una pagina por APDA y una por area CINE-F 13, enlazadas
entre si (cada APDA enlaza a sus areas CINE-F 13; cada area enlaza de vuelta a
su APDA y a las areas hermanas). Servirlo con cualquier servidor de archivos
estatico, por ejemplo `python -m http.server 8000 -d site`, o subiendo la
carpeta `site/` tal cual a un hosting estatico.

## Contenido

| Carpeta | Contiene |
|---|---|
| `build_dashboard.py` | Script que genera ambos dashboards + `build_dashboard_texto.md` (texto de la rama principal) + las paginas de `site/apda\|cine/` |
| `build_site.py` | Arma `site/index.html` (menu + Inicio + Decisiones) a partir de lo que dejo `build_dashboard.py`, y chequea que todos los links de `site/` resuelvan |
| `site/` | Sitio estatico unificado (generado, no versionado): `index.html`, `apda/*.html`, `cine/*.html`, `assets/plotly.min.js` |
| `Utils/` | Modulos compartidos que importa el builder (`utils`, `data_prep`, `variable_labels`, `report_html`, `archetypes_compat`) + `variable_glossary.yaml` |
| `Vector/` | Config, features y scores de la rama principal (por APDA) |
| `Archetypes/` | Config, scores, memberships y perfiles de arquetipos de la rama principal |
| `Cine/` | Config, features, scores, memberships, perfiles y texto de la rama por area CINE-F 13 |

Esta carpeta es autonoma: tiene su propia copia de `Utils/`, `Vector/`,
`Archetypes/` y `Cine/`, separada de la copia que usa `../Data processing/`
para regenerar los datos.

## Actualizar esta copia

Cuando cambien los datos o el script en `Proyecto APDAs`, volver a copiar los
mismos archivos desde ahi (no con `git`: las carpetas `*/outputs/` y
`*/data/` estan en `.gitignore` del repo original, asi que hay que copiarlas
del disco). Si se quiere regenerar estos numeros dentro de `APP UFT` en vez
de copiarlos de `Proyecto APDAs`, correr primero el pipeline de
`../Data processing/` (ver su README) y copiar sus salidas aca.
