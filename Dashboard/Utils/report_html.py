# =============================================================================
# Utils/report_html.py — Helpers compartidos para reportes HTML de análisis
# =============================================================================
# Extraído de "Inspection JCE/Exp_JCE_weighting_resultados.py" (primer reporte
# de este tipo) para reutilizar en reportes posteriores (ej. "Inspection
# Empleabilidad/"). La migración del script JCE para que también importe desde
# acá queda pendiente — no se tocó, para no arriesgar ese reporte ya validado.
#
# Uso típico:
#   from report_html import (fig_div, df_to_html_table, load_text_blocks,
#                             md_to_html, build_toc, render_page)
#   TEXTS = load_text_blocks("Inspection X/texto.md")
#   body_html = f"<section>{md_to_html(TEXTS['sec1_title'])}...</section>"
#   html = render_page(title, build_toc(body_html), body_html)
#   open("Inspection X/Reporte.html", "w", encoding="utf-8").write(html)
# =============================================================================

import re
import unicodedata

import pandas as pd
import plotly.io as pio

COLOR_BLUE = "#2a78d6"
COLOR_AQUA = "#1baf7a"
COLOR_YELLOW = "#eda100"
COLOR_VIOLET = "#4a3aa7"
COLOR_RED = "#e34948"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"


def fig_div(fig):
    return pio.to_html(
        fig, full_html=False, include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True},
    )


def sort_categoryarray(sub, col, y_col="ies_label"):
    """Orden del eje categórico por el valor de `col`, de mayor a menor de arriba
    hacia abajo."""
    return sub.sort_values(col, ascending=False)[y_col].tolist()


def sort_controls(sub, cols, labels, default_col, y_col="ies_label"):
    """Selector "Ordenar por" para figuras de barras horizontales que listan
    muchas categorías: reordena el eje categórico según el valor de cada serie
    sin recalcular las trazas (relayout de `categoryarray`).

    Devuelve `(updatemenus, annotations, categoryarray)` de una sola vez para que
    el orden inicial del eje y la opción marcada en el selector no se desalineen.
    """
    buttons = [
        dict(
            label=labels[c], method="relayout",
            args=[{
                "yaxis.categoryorder": "array",
                "yaxis.categoryarray": sort_categoryarray(sub, c, y_col),
            }],
        )
        for c in cols
    ]
    menus = [dict(
        type="dropdown", direction="down", buttons=buttons, active=cols.index(default_col),
        showactive=True, x=1.0, xanchor="right", y=1.0, yanchor="bottom", pad=dict(t=4, r=4),
        font=dict(size=10), bgcolor="#f5f4ef", bordercolor=COLOR_GRID,
    )]
    anns = [dict(
        text="Ordenar por:", showarrow=False, xref="paper", yref="paper",
        x=1.0, xanchor="right", y=1.075, yanchor="bottom", font=dict(size=10, color=COLOR_MUTED),
    )]
    return menus, anns, sort_categoryarray(sub, default_col, y_col)


def df_to_html_table(d, index=True, float_fmt="{:,.2f}"):
    return d.to_html(index=index, float_format=lambda v: float_fmt.format(v), border=0, classes="data-table")


def load_text_blocks(path):
    """Lee el .md de prosa del reporte y lo parte en bloques por `<!-- id: x -->`."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    parts = re.split(r"<!--\s*id:\s*(\w+)\s*-->\n", raw)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def _md_inline(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


_SLUGS_USADOS = {}


def _slugify(text):
    """Ancla estable (y única) para un heading, usada por el índice lateral."""
    txt = re.sub(r"<[^>]+>", "", text)
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii").lower()
    txt = re.sub(r"[^a-z0-9]+", "-", txt).strip("-") or "sec"
    n = _SLUGS_USADOS.get(txt, 0) + 1
    _SLUGS_USADOS[txt] = n
    return txt if n == 1 else f"{txt}-{n}"


def md_to_html(text, css_class=None):
    """Convierte un subconjunto simple de Markdown (headings #/##/###/####,
    párrafos, **negrita**, `code`) a HTML. Cada heading recibe un `id` para que
    el índice pueda enlazarlo. Si css_class está dado, se aplica al primer
    <p> generado (equivalente a <p class="...">)."""
    out, para = [], []

    def flush():
        if para:
            out.append("<p>" + _md_inline(" ".join(para)) + "</p>")
            para.clear()

    for line in (ln.strip() for ln in text.strip().splitlines()):
        if not line:
            flush()
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush()
            level = len(m.group(1))
            inner = _md_inline(m.group(2))
            out.append(f'<h{level} id="{_slugify(inner)}">{inner}</h{level}>')
        else:
            para.append(line)
    flush()
    html_frag = "\n".join(out)
    if css_class:
        html_frag = html_frag.replace("<p>", f'<p class="{css_class}">', 1)
    return html_frag


def build_toc(body):
    """Índice lateral a partir de los h2/h3 ya emitidos con `id` por md_to_html.
    Los h4 quedan fuera a propósito (suelen ser tarjetas de una grilla, no
    secciones de navegación)."""
    items = re.findall(r'<h([23])\b[^>]*\sid="([^"]+)"[^>]*>(.*?)</h\1>', body, flags=re.DOTALL)
    out, en_sub = [], False
    for level, anchor, texto in items:
        texto = re.sub(r"<[^>]+>", "", texto).strip()
        link = f'<a href="#{anchor}">{texto}</a>'
        if level == "3":
            if not en_sub:
                out.append('<ul class="toc-sub">')
                en_sub = True
            out.append(f"<li>{link}</li>")
        else:
            if en_sub:
                out.append("</ul></li>")
                en_sub = False
            elif out:
                out.append("</li>")
            out.append(f"<li>{link}")
    out.append("</ul></li>" if en_sub else "</li>")
    return "<ul>\n" + "\n".join(out) + "\n</ul>"


PAGE_CSS = """
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 1440px; margin: 0 auto; padding: 24px; color: #262624; }
  h1 { font-size: 1.6rem; border-bottom: 2px solid #e1e0d9; padding-bottom: 10px; }
  h2 { font-size: 1.25rem; margin-top: 48px; color: #1a1a18; }
  h3 { font-size: 1rem; color: #1a1a18; margin-top: 28px; }
  h4 { font-size: 0.95rem; color: #1a1a18; margin: 0 0 4px; border-left: 3px solid %(blue)s; padding-left: 8px; }
  section { margin-bottom: 40px; }
  p.desc { color: #57564f; max-width: 900px; line-height: 1.5; }
  table.data-table { border-collapse: collapse; margin: 16px 0; font-size: 0.85rem; }
  table.data-table th, table.data-table td { padding: 5px 10px; border-bottom: 1px solid #e1e0d9; text-align: right; }
  table.data-table th { background: #f5f4ef; text-align: right; }
  table.data-table td:first-child, table.data-table th:first-child { text-align: left; }
  .table-wrap { overflow-x: auto; }
  .callout { background: #f5f4ef; border-left: 3px solid %(blue)s; border-radius: 6px; padding: 12px 16px; margin: 16px 0; }
  /* índice lateral */
  .layout { display: flex; gap: 32px; align-items: flex-start; }
  main { min-width: 0; flex: 1 1 auto; }
  nav.toc { flex: 0 0 250px; position: sticky; top: 24px; max-height: calc(100vh - 48px); overflow-y: auto;
             font-size: 0.8rem; border-left: 2px solid #e1e0d9; padding-left: 14px; }
  nav.toc .toc-title { font-weight: 600; color: #1a1a18; text-transform: uppercase; letter-spacing: 0.04em;
                        font-size: 0.72rem; margin-bottom: 8px; }
  nav.toc ul { list-style: none; margin: 0; padding: 0; }
  nav.toc ul.toc-sub { padding-left: 12px; }
  nav.toc li { margin: 4px 0; line-height: 1.35; }
  nav.toc a { color: #57564f; text-decoration: none; }
  nav.toc a:hover { color: %(blue)s; }
  nav.toc a.active { color: %(blue)s; font-weight: 600; }
  /* grilla 2×2 de figuras por APDA */
  .apda-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; margin: 16px 0; }
  .apda-card { border: 1px solid #e1e0d9; border-radius: 10px; padding: 12px 14px; min-width: 0; }
  @media (max-width: 1100px) { .apda-grid { grid-template-columns: 1fr; } }
  @media (max-width: 1000px) {
    .layout { display: block; }
    nav.toc { position: static; max-height: none; margin-bottom: 28px; }
  }
""" % {"blue": COLOR_BLUE}

_TOC_HIGHLIGHT_JS = """
document.addEventListener("DOMContentLoaded", function() {
  var links = {};
  document.querySelectorAll("nav.toc a").forEach(function(a) {
    links[a.getAttribute("href").slice(1)] = a;
  });
  var headings = Array.prototype.filter.call(
    document.querySelectorAll("main h2[id], main h3[id]"),
    function(h) { return links[h.id]; }
  );
  var visibles = {};
  function marcar() {
    var activo = null;
    for (var i = 0; i < headings.length; i++) {
      if (visibles[headings[i].id]) { activo = headings[i].id; break; }
    }
    if (!activo) return;
    Object.keys(links).forEach(function(id) { links[id].classList.toggle("active", id === activo); });
  }
  var obs = new IntersectionObserver(function(entries) {
    entries.forEach(function(e) { visibles[e.target.id] = e.isIntersecting; });
    marcar();
  }, { rootMargin: "0px 0px -75% 0px" });
  headings.forEach(function(h) { obs.observe(h); });
});
"""


def render_page(title, toc_html, body_html, intro_html="", extra_head="", extra_script=""):
    """Ensambla la página completa: head (plotly CDN + CSS) + índice lateral
    sticky + cuerpo + resaltado de sección activa al hacer scroll."""
    plotly_js_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{title}</title>
{plotly_js_cdn}
{extra_head}
<style>{PAGE_CSS}</style>
</head>
<body>
{intro_html}
<div class="layout">
<nav class="toc"><div class="toc-title">Contenidos</div>
{toc_html}
</nav>
<main>
{body_html}
</main>
</div>
<script>
{_TOC_HIGHLIGHT_JS}
{extra_script}
</script>
</body>
</html>
"""


# =============================================================================
# DASHBOARD_CSS — shared with Dashboard/build_dashboard.py::render_page() and
# Dashboard/build_site.py's hub page, so the unified site (Dashboard/site/)
# looks the same everywhere without duplicating this block in two scripts.
# Extracted verbatim from build_dashboard.py's former inline <style> (Part A).
# =============================================================================
DASHBOARD_CSS = """
  body { font-family: Arial, Helvetica, sans-serif; margin: 0; padding: 0; background: #f7f8fa; color: #222; }
  header { background: #0f1420; color: white; padding: 18px 28px; }
  header h1 { margin: 0 0 4px 0; font-size: 20px; }
  header p { margin: 0; font-size: 13px; color: #a9b2c3; }
  .tabs { display: flex; flex-wrap: wrap; gap: 6px; padding: 14px 28px 0 28px; background: #f7f8fa; }
  .tab-btn { border: 1px solid #ccc; background: white; padding: 8px 16px; cursor: pointer;
              border-radius: 6px 6px 0 0; font-size: 13px; }
  .tab-btn.active { background: #0f1420; color: white; border-color: #0f1420; }
  .tabs a.tab-btn { text-decoration: none; display: inline-block; }
  .tabs.group-row { background: #eceff4; border-top: 1px solid #dde1e8; padding: 10px 28px; }
  .tabs .row-label { align-self: center; font-size: 12px; color: #666; margin-right: 2px; }
  .tab-content { display: none; padding: 20px 28px 40px 28px; }
  .tab-content.active { display: block; }
  .chart-block { background: white; border: 1px solid #e3e5e9; border-radius: 8px;
                   padding: 12px 16px; margin-bottom: 20px; }
  .chart-block h3 { margin-top: 0; font-size: 15px; color: #333; }
  .chart-block h3 a { color: inherit; }
  .note { color: #888; font-style: italic; font-size: 13px; }
  .subtab-buttons { display: flex; gap: 6px; margin-bottom: 10px; }
  .subtab-btn { border: 1px solid #ccc; background: #f0f1f3; padding: 6px 14px; cursor: pointer;
                 border-radius: 5px; font-size: 12px; }
  .subtab-btn.active { background: #3881BC; color: white; border-color: #3881BC; }
  .subtab-content { display: none; }
  .subtab-content.active { display: block; }
  .dim-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .dim-table th, .dim-table td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e3e5e9; vertical-align: top; }
  .dim-table th { color: #555; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }
  .dim-name { font-weight: bold; white-space: nowrap; width: 220px; }
  .var-list { margin: 0; padding-left: 18px; }
  .var-list li { margin-bottom: 2px; }
  .inv-flag { color: #d62828; font-size: 11px; font-style: italic; }
  .var-selector { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 10px; }
  .var-selector select { border: 1px solid #ccc; background: white; padding: 6px 10px;
                          border-radius: 5px; font-size: 12px; max-width: 320px; }
  .info-icon { display: inline-flex; align-items: center; justify-content: center;
                color: #3881BC; font-size: 15px; line-height: 1; cursor: help;
                position: relative; vertical-align: middle; }
  .info-icon .tooltip-box { visibility: hidden; opacity: 0; transition: opacity 0.15s;
                position: absolute; z-index: 60; bottom: 130%; left: 0;
                background: #0f1420; color: #e8ebf1; text-align: left; font-weight: normal;
                padding: 10px 12px; border-radius: 6px; font-size: 11.5px; line-height: 1.5;
                width: 300px; box-shadow: 0 3px 10px rgba(0,0,0,0.3); }
  .info-icon:hover .tooltip-box { visibility: visible; opacity: 1; }
  .info-icon .tooltip-box b { color: #7fc4ff; }
  .info-icon .tooltip-box i { color: #a9b2c3; font-style: normal; }
  .target-result { margin-top: 10px; overflow-x: auto; }
  .target-result:empty { margin-top: 0; }
  .dim-badge { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                margin-right: 6px; vertical-align: middle; }
  .chg-cell { display: flex; align-items: center; gap: 8px; white-space: nowrap; }
  .chg-bar-track { position: relative; width: 130px; height: 12px; background: #eee; border-radius: 2px; }
  .chg-bar { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 2px; }
  .chg-pct { font-size: 12px; color: #555; min-width: 42px; }
  .weights-header { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
  .settings-btn { border: 1px solid #3881BC; background: white; color: #3881BC; padding: 6px 14px;
                    cursor: pointer; border-radius: 5px; font-size: 12px; white-space: nowrap; }
  .settings-btn:hover { background: #eaf3fb; }
  .map-with-weights { display: flex; gap: 20px; align-items: flex-start; }
  .map-main { flex: 1 1 0; min-width: 0; }
  .weights-panel { display: none; flex-direction: column; gap: 16px; flex: 0 0 280px; }
  .weights-axis { min-width: 0; }
  .weights-axis h4 { margin: 0 0 6px 0; font-size: 13px; color: #333; }
  .weights-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .weights-table th, .weights-table td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #e3e5e9; }
  .weights-table input { width: 70px; border: 1px solid #ccc; border-radius: 4px; padding: 3px 6px; font-size: 12px; }
  .weights-total { margin-top: 6px; font-size: 12px; color: #555; }
  .reset-btn { border: 1px solid #ccc; background: #f0f1f3; padding: 6px 14px; cursor: pointer;
                border-radius: 5px; font-size: 12px; align-self: flex-start; }
  .reset-btn:hover { background: #e3e5e9; }
  .bi-heading { font-size: 17px; font-weight: bold; color: #222; padding: 16px 28px 0 28px; }
"""
