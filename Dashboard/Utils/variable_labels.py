# =============================================================================
# Utils/variable_labels.py — Showcase (display) names for glossary variables
# =============================================================================
# Thin, reusable helper over Utils/variable_glossary.yaml's `nombre_corto`
# field. Any script that needs to show a variable name to a person — not just
# Dashboard/ — should import this instead of re-implementing the fallback
# chain (nombre_corto -> nombre -> raw variable name).
#
# Usage:
#   from variable_labels import display_name, label_map
#   display_name("pct_jce_doctor")                  # -> "% JCE con doctorado"
#   label_map(["pct_jce_doctor", "unknown_var"])     # -> {"pct_jce_doctor": "% JCE con doctorado", "unknown_var": "unknown_var"}
# =============================================================================

import yaml

GLOSSARY_PATH = "Utils/variable_glossary.yaml"

ORIENT_LABEL = {
    "mayor_es_mejor": "Mayor es mejor ↑",
    "menor_es_mejor": "Menor es mejor ↓ (orientación invertida)",
}

_glossary_cache = None


def load_glossary():
    """Loads (and caches) Utils/variable_glossary.yaml."""
    global _glossary_cache
    if _glossary_cache is None:
        with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
            _glossary_cache = yaml.safe_load(f)
    return _glossary_cache


def display_name(var, glossary=None):
    """Showcase name for `var`: nombre_corto -> nombre -> `var` itself."""
    glossary = glossary if glossary is not None else load_glossary()
    entry = glossary.get(var)
    if not entry:
        return var
    return entry.get("nombre_corto") or entry.get("nombre") or var


def label_map(variables, glossary=None):
    """{var: display_name(var)} for an iterable of variable names."""
    glossary = glossary if glossary is not None else load_glossary()
    return {var: display_name(var, glossary) for var in variables}


def info_icon_html(var, glossary=None):
    """Ícono ⓘ con tooltip CSS (hover) mostrando la descripción de `var` desde
    Utils/variable_glossary.yaml. Cadena vacía si la variable no tiene entrada.

    Requiere el CSS .info-icon/.tooltip-box (ver Dashboard/build_dashboard.py)
    en la página que lo consume."""
    glossary = glossary if glossary is not None else load_glossary()
    entry = glossary.get(var)
    if not entry:
        return ""
    orient = ORIENT_LABEL.get(entry["orientacion"], entry["orientacion"])
    notas = f"<br><i>Notas:</i> {entry['notas']}" if entry.get("notas") else ""
    box = (
        f"<b>{entry['nombre']}</b><br>{entry['descripcion']}<br><br>"
        f"<i>Cálculo:</i> {entry['calculo']}<br>"
        f"<i>Fuente:</i> {entry['fuente']}<br>"
        f"<i>Unidad:</i> {entry['unidad']} · {orient}{notas}"
    )
    return f'<span class="info-icon">&#128712;<span class="tooltip-box">{box}</span></span>'
