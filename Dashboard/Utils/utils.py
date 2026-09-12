# =============================================================================
# utils_pca.py
# Funciones auxiliares para el Modelo de Posicionamiento Institucional (UFT)
# =============================================================================

import unicodedata
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# tab
# =============================================================================

def tab(var1, var2=None, missing=False, pct='col'):
    s1 = var1.fillna('(missing)') if missing else var1.dropna()

    if var2 is None:
        counts = s1.value_counts(dropna=True).sort_index()
        total  = counts.sum()
        pct_s  = counts / total * 100
        cum    = pct_s.cumsum()

        name  = str(var1.name) if var1.name else 'var'
        val_w = max((len(str(v)) for v in counts.index), default=5)
        val_w = max(val_w, len(name), 5)
        sep   = f"{'─'*val_w}  {'─'*8}  {'─'*7}  {'─'*8}"

        print(f"\n{name:<{val_w}}  {'Freq':>8}  {'Pct':>7}  {'Cum Pct':>8}")
        print(sep)
        for val, freq, p, cp in zip(counts.index, counts.values, pct_s.values, cum.values):
            print(f"{str(val):<{val_w}}  {freq:>8,}  {p:>6.1f}%  {cp:>7.1f}%")
        print(sep)
        print(f"{'Total':<{val_w}}  {total:>8,}  {'100.0%':>8}\n")

    else:
        s2 = var2.fillna('(missing)') if missing else var2[s1.index]

        ct    = pd.crosstab(s1, s2, margins=True, margins_name='Total')
        norm  = 'columns' if pct == 'col' else 'index'
        pct_t = pd.crosstab(s1, s2, normalize=norm) * 100

        label = 'Col %' if pct == 'col' else 'Row %'
        print(f"\n── Counts ── {var1.name} × {var2.name}")
        print(ct.to_string())
        print(f"\n── {label} ──")
        print(pct_t.round(1).to_string())
        print()



# =============================================================================
# Inspect
# =============================================================================
def ins(df, x):
    print(df[x].unique())


# =============================================================================
# Norm ies
# =============================================================================

def normalizar_nombre_ies(nombre: str) -> str:
    nombre = str(nombre).strip().lower()
    nombre = unicodedata.normalize("NFKD", nombre)
    nombre = ''.join(c for c in nombre if not unicodedata.combining(c))
    return nombre


# =============================================================================
# Norm carrera
# =============================================================================

def normalizar_nombre_carrera(nombre: str) -> str:
    """Normaliza nombres de carrera para el match SIES Buscador ↔ Oferta Académica.
    Más agresivo que normalizar_nombre_ies: elimina puntuación y colapsa espacios,
    porque los nombres de carrera difieren en comas, paréntesis y NBSP entre fuentes."""
    s = unicodedata.normalize('NFKD', str(nombre).strip().lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9ñ ]', ' ', s.replace('\xa0', ' '))
    return re.sub(r'\s+', ' ', s).strip()


# =============================================================================
# 1. NORMALIZACIÓN
# =============================================================================

def normalizar_nombre_columna(col: str) -> str:
    """
    Normaliza un nombre de columna: minúsculas, sin tildes, sin espacios ni
    caracteres especiales. Reemplaza '%' por 'porc'.

    Fuente: bloque CONFIGURACIÓN, no_monto_nuevos_ind2.py
    Input:  str — nombre de columna original
    Output: str — nombre normalizado
    """
    col = str(col).strip().lower()
    col = unicodedata.normalize("NFKD", col)
    col = ''.join(c for c in col if not unicodedata.combining(c))
    col = col.replace(" ", "_")
    col = col.replace("–", "_").replace("—", "_")
    col = col.replace("°", "")
    col = col.replace(".", "_")
    col = col.replace("%", "porc")
    return col



# =============================================================================
# 2. MAPAS DE REFERENCIA
# =============================================================================

# Diccionario canónico IES → sigla (nombres ya normalizados, sin tildes)
abreviaciones_ies: dict[str, str] = {
    "pontificia universidad catolica de chile": "PUC",
    "pontificia universidad catolica de valparaiso": "PUCV",
    "universidad adolfo ibanez": "UAI",
    "universidad alberto hurtado": "UAH",
    "universidad andres bello": "UNAB",
    "universidad nacional andres bello": "UNAB",
    "universidad arturo prat": "UNAP",
    "universidad austral de chile": "UACH",
    "universidad autonoma de chile": "UAUTON",
    "universidad bernardo o'higgins": "UBO",
    "universidad catolica de la santisima concepcion": "UCSC",
    "universidad catolica de temuco": "UCT",
    "universidad catolica del maule": "UCM",
    "universidad catolica del norte": "UCN",
    "universidad central de chile": "UCEN",
    "universidad de antofagasta": "UANTOF",
    "universidad de chile": "UCH",
    "universidad de concepcion": "UDEC",
    "universidad de la frontera": "UFRO",
    "universidad de la serena": "ULS",
    "universidad de las americas": "UDLA",
    "universidad de los andes": "UANDES",
    "universidad de los lagos": "ULAGOS",
    "universidad de o'higgins": "UOH",
    "universidad de magallanes": "UMAG",
    "universidad de playa ancha de ciencias de la educacion": "UPLA",
    "universidad de santiago de chile": "USACH",
    "universidad de talca": "UTALCA",
    "universidad de tarapaca": "UTA",
    "universidad de valparaiso": "UV",
    "universidad del bio-bio": "UBB",
    "universidad del desarrollo": "UDD",
    "universidad diego portales": "UDP",
    "universidad finis terrae": "UFT",
    "universidad mayor": "UMAYOR",
    "universidad san sebastian": "USS",
    "universidad tecnica federico santa maria": "UFSM",
}


def slug(nombre: str) -> str:
    """Nombre de grupo (APDA o area CINE) -> slug ascii-only para URLs y nombres
    de archivo del sitio unificado (Dashboard/build_site.py) — a diferencia de
    build_dashboard.py::safe_id, que solo reemplaza espacios/'/' y deja comas/
    acentos (sirve para ids del DOM, no para rutas)."""
    return re.sub(r'[^a-z0-9]+', '-', normalizar_nombre_ies(nombre)).strip('-')


