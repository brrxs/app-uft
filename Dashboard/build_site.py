# =============================================================================
# Dashboard/build_site.py — Assembles Dashboard/site/index.html (the unified
# hub: main menu + Inicio APDA + Inicio CINE note + Decisiones) from the
# fragments and per-group pages build_dashboard.py already wrote, then checks
# every relative href under site/ resolves to a real file.
#
# Run after both branch builds (from APP UFT/Dashboard):
#   python build_dashboard.py
#   python build_dashboard.py Cine/config.yaml
#   python build_site.py
# =============================================================================
import glob
import os
import re
import sys

import yaml

sys.path.insert(0, "Utils")
from report_html import DASHBOARD_CSS
from utils import slug

with open("Cine/config.yaml", "r", encoding="utf-8") as f:
    PARENT_GROUPS = (yaml.safe_load(f).get("dashboard") or {}).get("parent_groups") or {}


def _read(path):
    if not os.path.exists(path):
        print(f"Warning: missing {path} — run build_dashboard.py (both branches) first.")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


inicio_apda = _read("site/_fragments/inicio-apda.html")
inicio_cine = _read("site/_fragments/inicio-cine.html")
# decisiones_section starts as an inactive tab (class="tab-content", no JS here
# to switch to it) — force it visible on the hub, same trick nowhere else needed
# since every other embedded fragment is already "active" in its source.
decisiones = _read("site/_fragments/decisiones.html").replace(
    'class="tab-content"', 'class="tab-content active"', 1
)

menu_cards = []
for apda, areas in PARENT_GROUPS.items():
    area_links = "".join(f'<a class="tab-btn" href="cine/{slug(a)}.html">{a}</a>' for a in areas)
    menu_cards.append(f"""
      <div class="chart-block">
        <h3><a href="apda/{slug(apda)}.html">{apda}</a></h3>
        <div class="tabs" style="padding:0;">{area_links}</div>
      </div>""")

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Dashboard — Proyecto APDAs</title>
<style>
{DASHBOARD_CSS}
</style>
</head>
<body>
<header>
  <h1>Dashboard — Proyecto APDAs</h1>
  <p>Visualización preliminar de resultados, no compartir ni descargar figuras.</p>
</header>
<div class="tabs">
  <a class="tab-btn active" href="index.html">Inicio</a>
  <a class="tab-btn" href="#decisiones">Decisiones</a>
</div>
<div class="tab-content active" id="tab-menu">
  <div class="chart-block">
    <h3>Menú principal</h3>
    <p class="note">Cada APDA enlaza a su propio dashboard completo; debajo de cada una,
      las áreas CINE-F 13 que la componen enlazan al dashboard de esa área. Desde
      cualquier página de APDA o de área se puede volver aquí o saltar a las páginas
      relacionadas.</p>
  </div>
  {''.join(menu_cards)}
</div>
{inicio_apda}
<div class="tab-content active" id="tab-inicio-cine">{inicio_cine}</div>
<section id="decisiones">{decisiones}</section>
</body>
</html>
"""

os.makedirs("site", exist_ok=True)
with open("site/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Saved: site/index.html")

# ── Link check: every relative href under site/**/*.html resolves to a file ─
href_re = re.compile(r'href="([^"#][^"]*)"')
pages = glob.glob("site/**/*.html", recursive=True)
missing = []
for page in pages:
    base = os.path.dirname(page)
    with open(page, "r", encoding="utf-8") as f:
        content = f.read()
    for href in href_re.findall(content):
        if re.match(r'^[a-z][a-z0-9+.-]*:', href, re.I) or href.startswith("//"):
            continue  # absolute URL (http(s), mailto, etc.) — not ours to check
        file_part = href.split("#", 1)[0]  # strip any #anchor before checking
        if not file_part:
            continue  # pure same-page anchor, e.g. href="#decisiones"
        target = os.path.normpath(os.path.join(base, file_part))
        if not os.path.exists(target):
            missing.append(f"{page} -> {href}")

if missing:
    print(f"\n{len(missing)} broken link(s):")
    for m in missing:
        print(f"  {m}")
    sys.exit(1)
print(f"Link check OK ({len(pages)} pages under site/)")
