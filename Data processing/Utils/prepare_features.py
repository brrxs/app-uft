# =============================================================================
# prepare_features.py — Build the IES×APDA feature table (shared by
# Vector/00_prepare_data.py and Archetypes/00_prepare_data.py)
# =============================================================================
# Reads dfs/master_dataset.csv via data_prep.py and writes
# <out_dir>/<out_name>. Both branches call this with their own
# config.yaml and data/ dir, producing structurally identical tables built
# independently, so neither branch has a runtime dependency on the other.
#
# group_col reagrupa el universo por otra columna en vez de por APDA (ver
# load_universe en data_prep.py). Lo usa Cine/00_prepare_data.py con
# 'cine-f 13 area' para construir la misma tabla a grano IES x area CINE-F 13.
# =============================================================================

import os
import yaml

from data_prep import (
    load_universe, build_inventory, feature_sets, aggregate_ies_apda_rich,
    add_parity_scores, add_acred_scores, add_empleabilidad_ponderada,
    add_prop_matricula_posgrado_apda, add_puntaje_seleccion,
    add_pct_tes_particular_pagado_apda,
)


def main(config_path, out_dir, out_name="ies_apda_features.csv", group_col=None):
    os.makedirs(out_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as _f:
        cfg = yaml.safe_load(_f)

    acred_min_filter = cfg["acred_min_filter"]

    df, u = load_universe(
        path="Data/master_dataset.csv",
        matricula_filter=True,
        acred_filter=False,
        acred_min=acred_min_filter,
        group_col=group_col,
    )

    inv = build_inventory(df, u)
    _, feat_all = feature_sets(inv)
    ies_apda = aggregate_ies_apda_rich(u, feat_all)
    ies_apda = add_parity_scores(u, ies_apda)
    ies_apda = add_acred_scores(u, ies_apda)
    ies_apda = add_empleabilidad_ponderada(u, ies_apda)
    ies_apda = add_prop_matricula_posgrado_apda(u, ies_apda)
    ies_apda = add_puntaje_seleccion(u, ies_apda)
    ies_apda = add_pct_tes_particular_pagado_apda(u, ies_apda)

    grupo = group_col or "apda"
    out_path = f"{out_dir}/{out_name}"
    ies_apda.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}  shape={ies_apda.shape}  "
          f"({u['apda'].nunique()} grupos de '{grupo}', {len(ies_apda)} celdas)")
