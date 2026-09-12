# =============================================================================
# data_prep.py — Módulo compartido de preparación de datos y agregación
# =============================================================================
# Usado por Vector/00_prepare_data.py y Archetypes/00_prepare_data.py como
# único punto de acceso a dfs/master_dataset.csv.
# API pública:
#   Constants: TARGET, COV_MIN, EXCLUDE_FAMILIES, EXCLUDE, INST_COL,
#              NOMBRE_COL, PROG_EXCL_PREFIXES, FIRST_PREFIXES
#   load_universe(path, matricula_filter, acred_filter, acred_min) -> (df, u)
#   build_inventory(df, u) -> inv
#   feature_sets(inv) -> (feat_prog, feat_all)
#   aggregate_ies_apda_rich(u, feat_all) -> agg_df
#   add_parity_scores(u, agg_df) -> agg_df
#   add_acred_scores(u, agg_df) -> agg_df
#   add_empleabilidad_ponderada(u, agg_df) -> agg_df
#   add_prop_matricula_posgrado_apda(u, agg_df) -> agg_df
#   add_puntaje_seleccion(u, agg_df) -> agg_df
#   add_pct_tes_particular_pagado_apda(u, agg_df) -> agg_df
# =============================================================================

# %% Imports

import unicodedata
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# %% Constantes

TARGET           = 'Años acreditación (30 de octubre de 2025)'
COV_MIN          = 0.60
EXCLUDE_FAMILIES = {'id_code', 'flag'}
EXCLUDE          = {'versión', 'régimen', 'área actual', 'is_post', 'clas_est joven'}

# Indicadores con cobertura intrínsecamente baja (NaN fuera de su ámbito por
# diseño, ej. pct_jce_especialidad sólo aplica al APDA de Salud y bienestar,
# ver derivación más abajo;
# empleabilidad_*/continuidad_estudios sólo cubren Pregrado y sólo ~27% de
# esos programas, dado que la fuente SIES no publica sede/jornada/versión y
# el match es a nivel institución×programa — ver "Inspection Empleabilidad/";
# total matrícula primer año/prop_matricula_vacantes_año1 sólo aplican a
# Pregrado desde 2026-08-17 —ver máscara en load_universe— y su cobertura
# dentro de Pregrado (~58%) ya quedaba por debajo de COV_MIN incluso antes
# del filtro) que deben incluirse en feat_all igual, saltándose el filtro de
# cobertura ≥ COV_MIN. acred_posgrado_ajustado/acred_doctorado_ajustado/
# acred_especialidad_medica no necesitan estar aquí: se calculan después de
# build_inventory (ver add_acred_scores), así que nunca pasan por este filtro.
# promedio_puntaje/minimo_puntaje/puntaje_corte (CNED) tampoco: su cobertura a
# nivel programa (22-29%) queda muy por debajo de COV_MIN incluso tras el
# match ampliado a I/S/C, así que se retiran del pipeline genérico y se
# calculan aparte como promedio_puntaje_apda/puntaje_corte_apda — ver
# add_puntaje_seleccion.
FORCE_INCLUDE    = {
    'pct_jce_especialidad',
    'sobreduracion_meses', 'sobreduracion_semestres',
    'empleabilidad_1er_año', 'empleabilidad_2do_año', 'continuidad_estudios',
    'total matrícula primer año', 'prop_matricula_vacantes_año1',
}
INST_COL         = 'clasificación institución nivel 1'
NOMBRE_COL       = 'nombre institución'

# Prefijos excluidos del modelo programa-nivel (no varían dentro de IES o son pesos)
PROG_EXCL_PREFIXES = ('anid_', 'scimago_', 'scival_', 'w_mat', 'w_doc', 'w_post',
                       'w_academicos_39', 'w_jce_doctor', 'w_jce_magister', 'w_jce_especialidad',
                       'pct_jce_', 'pct_academicos_39', 'jce_por_estudiantes')

# Prefijos que son constantes dentro de IES×APDA → 'first' en la agregación
# (anid_/scimago_/scival_ y pct_jce_*/pct_academicos_39/jce_por_estudiantes,
# constante por IES; w_mat/w_doc/w_post/w_jce_*/w_academicos_39 varían dentro
# de IES vía _allocator → 'mean'). pct_jce_especialidad es la excepción: es
# NaN fuera del APDA de Salud y bienestar (ver derivación arriba), así que
# 'first' allí resuelve a NaN en vez de al valor institucional.
FIRST_PREFIXES = ('anid_', 'scimago_', 'scival_', 'pct_jce_', 'pct_academicos_39', 'jce_por_estudiantes')

# Prefijos agregados por media ponderada por matrícula en vez de media simple.
# Necesario porque el match de empleabilidad_*/continuidad_estudios se hace a
# nivel institución×programa (la fuente SIES no publica sede/jornada/versión):
# un mismo valor se replica sobre varias filas de programa (sedes/jornadas/
# versiones distintas dentro del mismo programa; fan-out medio 2.4, máx 20),
# y sin ponderar esas copias inflarían el peso de programas muy fraccionados
# dentro de la celda IES×APDA. Ver "Inspection Empleabilidad/".
WEIGHTED_FEATS_PREFIXES = ('empleabilidad_', 'continuidad_estudios')
WEIGHT_COL = 'total matrícula'

# Variables que deben sumarse (no promediarse) al colapsar a IES×APDA: son
# headcounts institucionales, no características típicas de un programa. Sin
# esto, 'total matrícula primer año' quedaba como sum(programas)/n_programs
# (media), deflactando el valor real bajo comparaciones con muchos programas.
SUM_FEATS = {'total matrícula primer año'}

# empleabilidad_pond_1er_año/2do_año: alternativa a empleabilidad_1er_año/2do_año
# ponderada por titulados de la carrera genérica (en vez de por matrícula del
# programa). Motivo: empleabilidad mide egresados, no matriculados — un
# programa grande con pocos titulados no debería pesar más que uno chico que
# titula mucho. dfs/catalogos/titulados_generica.csv (Exp_Titulados_Generica.py)
# trae el conteo de titulados por (cod_inst, área carrera genérica) en dos
# ventanas de 3 años, cada una alineada a la cohorte que mide la respectiva
# variable de empleabilidad. Ver add_empleabilidad_ponderada.
EMP_POND_SPECS = {
    'empleabilidad_pond_1er_año': ('empleabilidad_1er_año', 'titulados_gen_emp1'),
    'empleabilidad_pond_2do_año': ('empleabilidad_2do_año', 'titulados_gen_emp2'),
}
# Institución × programa, sin sede/jornada/versión — el grano real del
# Buscador de Empleabilidad SIES (ver comentario en Exp_SIES.py). Colapsa el
# fan-out sede/jornada/versión antes de ponderar, para no contar el mismo
# titulado varias veces dentro de una celda IES×APDA.
EMP_UNIT_KEYS = ['cod_inst', 'cod_carrera']
EMP_GEN_COL = 'área carrera genérica'

# Nombres de grupo que corresponden al ámbito "Salud". El APDA se llama
# 'Salud y bienestar' y el área CINE-F 13 equivalente 'Salud y Bienestar'
# (difieren sólo en la mayúscula de Bienestar), así que los indicadores
# acotados a Salud deben reconocer las dos. Ver load_universe(group_col=...).
SALUD_GROUPS = ('Salud y bienestar', 'Salud y Bienestar')

# %% Helpers

def _norm(s):
    s = str(s).strip().lower()
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c))


def _family(col):
    c = _norm(col)
    if 'matricula' in c:                                       return 'matricula'
    if 'jce' in c or 'academico' in c:                        return 'faculty_jce'
    if 'rango de edad' in c or 'edad' in c:                   return 'edad'
    if 'region' in c:                                          return 'region'
    if c.startswith('tes') or ' tes' in c or 'establecimiento' in c: return 'tes'
    if 'ponderacion' in c:                                     return 'ponderacion'
    if 'arancel' in c or 'costo' in c or 'monto' in c or 'valor' in c: return 'costo'
    if 'duracion' in c:                                        return 'duracion'
    if 'acreditacion' in c or 'acred' in c:                   return 'acreditacion'
    if 'ret_' in c or 'retencion' in c:                       return 'retencion'
    if c.startswith('cod') or 'codigo' in c or 'demre' in c or c == 'ano': return 'id_code'
    if c.startswith('flag'):                                   return 'flag'
    return 'other'

# %% Función 1: cargar y preparar universo

def load_universe(
    path='Data/master_dataset.csv',
    matricula_filter=True,
    acred_filter=False,
    acred_min=5,
    group_col=None,
):
    """Lee el CSV, coerce numérica, construye derivados y filtra universo.

    `group_col`: nombre de una columna de agrupación alternativa (ej.
    'cine-f 13 área'). Si se pasa, reemplaza el contenido de 'apda' por el de
    esa columna y guarda el valor original en 'apda_padre'. Todo el pipeline
    aguas abajo agrupa por 'apda' sin nombrar APDAs concretas, así que esto
    basta para correrlo completo sobre otra taxonomía. Las filas cuyo 'apda'
    es NaN o 'No aplica' se dejan intactas, para que el filtro de universo de
    más abajo las siga descartando igual.

    Retorna (df, u):
        df  — DataFrame completo tras coerción y derivados
        u   — subconjunto filtrado (universo activo) con columna ies_norm
    """
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    print(f"master_dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # ── Coerción numérica genérica (igual que A) ───────────────────────────
    for _c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[_c]):
            _coerced = pd.to_numeric(df[_c], errors='coerce')
            _nn = df[_c].notna().sum()
            if _nn > 0 and _coerced.notna().sum() >= 0.8 * _nn:
                df[_c] = _coerced

    # ── Reagrupación opcional ──────────────────────────────────────────────
    # Antes del bloque de derivadas, para que las variables que se calculan
    # por grupo (arancel_anual_index) o acotadas a un grupo
    # (pct_jce_especialidad) usen la agrupación nueva y no la APDA.
    if group_col:
        _sin_apda = df['apda'].isna() | (df['apda'] == 'No aplica')
        df['apda_padre'] = df['apda']
        df['apda'] = df['apda'].where(_sin_apda, df[group_col])

    # Indicadores binarios derivados (igual que A)
    acred_prog = df['acreditación carrera o programa']
    df['acred_bin'] = (
        acred_prog.notna()
        & ~acred_prog.str.contains('no acreditada|sin acreditación', case=False, na=False)
    ).astype(float)

    df['is_post'] = (df['nivel global'].astype(str).str.lower() != 'pregrado').astype(float)

    # ── Nuevas variables derivadas a nivel programa ────────────────────────
    df['vacantes_año1'] = df[['vacantes semestre uno', 'vacantes semestre dos']].sum(
        axis=1, min_count=1
    )

    # Acceso Primer año (dimensión de Vector/Archetypes config.yaml) sólo debe
    # reflejar acceso a Pregrado: sin este filtro, celdas IES×APDA cuyos únicos
    # programas son de Posgrado/Postítulo (ej. UFSM en "Artes, humanidades y
    # cultura", donde comparte celda con Arquitectura de Pregrado) mezclan
    # matrícula de posgrado, que no compite por las mismas vacantes de primer
    # año, con la de pregrado.
    df['total matrícula primer año'] = df['total matrícula primer año'].where(
        df['nivel global'] == 'Pregrado'
    )
    df['prop_matricula_vacantes_año1'] = (
        df['total matrícula primer año'] / df['vacantes_año1']
    ).replace([np.inf, -np.inf], np.nan)

    # acred_posgrado_ajustado/acred_doctorado_ajustado/acred_especialidad_medica
    # se calculan a nivel de celda IES×APDA en add_acred_scores (después de
    # aggregate_ies_apda_rich), no aquí a nivel de programa — su universo
    # combina dedup por cod_carrera y el filtro de vigencia, que no tiene
    # sentido aplicar fila por fila en el frame programa-nivel.

    # pct_jce_especialidad: JCE en especialidad médica u odontológica sólo es
    # un indicador relevante para el APDA de Salud — se anula fuera de ese APDA
    # para que no contamine el puntaje de IES en Artes/Ciencias/Educación con
    # un ratio que no tiene relación con esas áreas (ver "JCE_esp médicas").
    df['pct_jce_especialidad'] = df['pct_jce_especialidad'].where(
        df['apda'].isin(SALUD_GROUPS)
    )

    df['pct_matricula_hombres'] = (
        df['total matrícula hombres'] / df['total matrícula']
    ).replace([np.inf, -np.inf], np.nan)
    df['pct_matricula_mujeres'] = (
        df['total matrícula mujeres'] / df['total matrícula']
    ).replace([np.inf, -np.inf], np.nan)

    # arancel_anual_index: normalizado por apda, media→0, rango→[-1, 1]
    _g = df.groupby('apda')['arancel anual']
    _mean, _min, _max = _g.transform('mean'), _g.transform('min'), _g.transform('max')
    _above = df['arancel anual'] >= _mean
    _idx = pd.Series(np.nan, index=df.index)
    _idx[_above] = ((df['arancel anual'] - _mean) / (_max - _mean))[_above]
    _idx[~_above] = ((df['arancel anual'] - _mean) / (_mean - _min))[~_above]
    df['arancel_anual_index'] = _idx.replace([np.inf, -np.inf], 0.0)

    # duracion_real / duracion_teorica: duración total vs. duración de la fase lectiva.
    # brecha_duracion es siempre >= 0 (duración titulación nunca es negativa): mide
    # demora respecto a lo teórico, no puede detectar programas que terminan antes.
    df['duracion_real'] = df['duración total']
    df['duracion_teorica'] = df['duración estudios']
    df['brecha_duracion'] = df['duracion_real'] - df['duracion_teorica']

    # sobreduracion: exceso de duración observada (titulados, empírica) vs.
    # duración nominal declarada en la oferta ("duración total", en semestres).
    # duracion_titulados_mean viene en años -> se pasa a semestres (*2) antes de
    # restar. Sólo se calcula para niveles de carrera donde el concepto de
    # titulación aplica de forma comparable; se excluyen Bachillerato, Profesional
    # Sin Licenciatura y Técnico de Nivel Superior (resto de niveles -> NaN).
    _nivel_excl_sobreduracion = [
        'Bachillerato, Ciclo Inicial o Plan Común',
        'Profesional Sin Licenciatura',
        'Técnico de Nivel Superior',
    ]
    _nivel_ok_sobreduracion = ~df['nivel carrera'].isin(_nivel_excl_sobreduracion)
    _sobreduracion_semestres = (
        df['duracion_titulados_mean'] * 2 - df['duración total']
    ).where(_nivel_ok_sobreduracion)
    df['sobreduracion_semestres'] = _sobreduracion_semestres
    df['sobreduracion_meses'] = _sobreduracion_semestres * 6

    # ── Universe base (igual que A) ────────────────────────────────────────
    base_mask = (
        (df[INST_COL] == 'Universidades')
        & (df['apda'].notna())
        & (df['apda'] != 'No aplica')
    )

    mask = base_mask.copy()
    if matricula_filter:
        mask &= (df['flag_matricula'] == 1)
    if acred_filter:
        mask &= (df[TARGET] >= acred_min)

    u = df[mask].copy()
    print(f"\nUniverso activo: {len(u):,} programas | "
          f"{u[NOMBRE_COL].nunique()} IES | {u['apda'].nunique()} APDAs")

    if 'ies_norm' not in u.columns:
        u['ies_norm'] = u[NOMBRE_COL].map(_norm)

    return df, u

# %% Función 2: inventario de indicadores

def build_inventory(df, u):
    """Construye inventario de indicadores numéricos sobre el universo u.

    Retorna inv — DataFrame con columnas:
        indicator, family, coverage_pct, std, inst_constant, relevant
    """
    num_cols = df.select_dtypes(include='number').columns.tolist()

    inv_rows = []
    for col in num_cols:
        s_u = u[col]
        inv_rows.append({
            'indicator':     col,
            'family':        _family(col),
            'coverage_pct':  round(s_u.notna().mean() * 100, 1),
            'std':           s_u.std(ddof=0),
            # inst_constant: mismo valor para todos los programas de una IES?
            'inst_constant': bool(u.groupby(NOMBRE_COL)[col].nunique(dropna=True).max() <= 1),
        })

    inv = pd.DataFrame(inv_rows)
    inv['relevant'] = (
        (inv['coverage_pct'] >= COV_MIN * 100) | inv['indicator'].isin(FORCE_INCLUDE)
    ) & (inv['std'] > 0)
    return inv

# %% Función 3: construir feature sets

def feature_sets(inv):
    """Deriva los dos feature sets a partir del inventario.

    Retorna (feat_prog, feat_all):
        feat_prog — features para modelo programa-nivel (excluye inst_constant y prefijos)
        feat_all  — features para modelo agregado IES×APDA (conjunto completo)
    Ambas listas excluyen TARGET.
    """
    # feat_prog: sólo variables que varían dentro de IES
    feat_prog = inv.loc[
        inv['relevant']
        & ~inv['inst_constant']
        & ~inv['family'].isin(EXCLUDE_FAMILIES)
        & ~inv['indicator'].isin(EXCLUDE)
        & ~inv['indicator'].apply(
            lambda c: any(c.startswith(p) for p in PROG_EXCL_PREFIXES)
        ),
        'indicator'
    ].tolist()
    feat_prog = [c for c in feat_prog if c != TARGET]

    # feat_all: conjunto completo (incluye anid_*, scimago_*, pesos JCE)
    feat_all = inv.loc[
        inv['relevant']
        & ~inv['family'].isin(EXCLUDE_FAMILIES)
        & ~inv['indicator'].isin(EXCLUDE)
        & ~inv['indicator'].isin([TARGET]),
        'indicator'
    ].tolist()

    return feat_prog, feat_all

def prepare_features(X, strategy='median'):
    """Median-impute then z-standardize X (DataFrame or ndarray). Returns ndarray."""
    X_arr = X.values if hasattr(X, 'values') else X
    X_arr = SimpleImputer(strategy=strategy).fit_transform(X_arr)
    return StandardScaler().fit_transform(X_arr)


def normalize_variables_0_100(agg, variables, direction=None, apda_col='apda',
                               norm_method='zscore_within_apda', verbose=True):
    """Normaliza una lista de variables a 0-100.

    `zscore_within_apda`: z-score dentro de cada `apda_col`, reescalado
    globalmente a 0-100; `minmax`: min-max global. En ambos casos aplica antes
    `direction.get(var, 1)` (1 o -1) para que "mayor score = mejor" valga
    también para variables cuya semántica cruda va al revés (ej. costo, año
    de inicio de un programa).

    Usado por `compute_dimension_scores` y por Dashboard/build_dashboard.py
    (selector de variable individual) — único punto de normalización de
    variables a 0-100 en el proyecto.

    Retorna {var: pd.Series} solo para las variables presentes en `agg.columns`.
    """
    direction = direction or {}
    all_vars = list(dict.fromkeys(variables))
    missing_vars = [v for v in all_vars if v not in agg.columns]
    if missing_vars and verbose:
        print(f"\nWarning: {len(missing_vars)} configured variable(s) not found in data:")
        for v in missing_vars:
            print(f"  {v}")
    all_vars = [v for v in all_vars if v in agg.columns]
    if verbose:
        print(f"\n{len(all_vars)} configured variables found in data")

    inverted = [v for v in all_vars if direction.get(v, 1) < 0]
    if inverted and verbose:
        print(f"Inverted (direction=-1, lower raw value = higher score): {inverted}")

    def _zscore_group(s):
        sd = s.std(ddof=0)
        return s * 0.0 if (pd.isna(sd) or sd == 0) else (s - s.mean()) / sd

    norm_cols = {}
    if verbose:
        print(f"\nNormalizing ({norm_method})...")
    for var in all_vars:
        raw = pd.to_numeric(agg[var], errors='coerce') * direction.get(var, 1)

        if norm_method == 'zscore_within_apda':
            z = agg.assign(**{var: raw}).groupby(apda_col)[var].transform(_zscore_group)
            z_min, z_max = z.min(), z.max()
            if z_max > z_min:
                norm_cols[var] = (z - z_min) / (z_max - z_min) * 100.0
            else:
                norm_cols[var] = pd.Series(50.0, index=z.index)
        else:
            v_min, v_max = raw.min(), raw.max()
            if v_max > v_min:
                norm_cols[var] = (raw - v_min) / (v_max - v_min) * 100.0
            else:
                norm_cols[var] = pd.Series(50.0, index=raw.index)

    return norm_cols


def compute_dimension_scores(agg, dimensions, id_cols, target_col=None,
                              norm_method='zscore_within_apda', apda_col='apda',
                              direction=None, verbose=True):
    """Normaliza variables configuradas a 0-100 y las agrega en dimension scores.

    Usado por Vector/01_vectors.py, Dashboard/build_dashboard.py y por
    cualquier otra rama que consuma una tabla IES×APDA y quiera el mismo
    espacio de dimensiones curadas — incluyendo Sensitivity/, que la llama
    miles de veces por corrida (ver `run_full_pipeline` más abajo), de ahí
    `verbose=False` para no inundar stdout.

    Para cada dimensión en `dimensions` (dict: nombre -> {"variables": {var: peso}}):
      1. normaliza cada variable a 0-100 (ver `normalize_variables_0_100`).
      2. calcula un promedio ponderado NaN-aware (una fila que falta una variable
         igual recibe un puntaje de las variables que sí tiene, en vez de colapsar
         a NaN — importante para indicadores de ámbito parcial como acred_posgrado_ajustado).

    Retorna (dim_scores, valid_dims):
        dim_scores — DataFrame con id_cols [+ target_col] + una columna por dimensión
                     con al menos una variable presente
        valid_dims — lista de nombres de dimensión efectivamente calculadas
    """
    direction = direction or {}
    all_vars = list(dict.fromkeys(
        v for dim in dimensions.values() for v in dim['variables']
    ))
    norm_cols = normalize_variables_0_100(
        agg, all_vars, direction=direction, apda_col=apda_col, norm_method=norm_method,
        verbose=verbose,
    )

    if verbose:
        print("\nComputing dimension scores...")
    keep_cols = list(id_cols)
    if target_col and target_col in agg.columns:
        keep_cols = keep_cols + [target_col]
    dim_scores = agg[keep_cols].copy()

    valid_dims = []
    for dim_name, dim_cfg in dimensions.items():
        vars_in_dim = {v: w for v, w in dim_cfg['variables'].items() if v in norm_cols}
        if not vars_in_dim:
            if verbose:
                print(f"  Skipping '{dim_name}': no valid variables")
            continue
        # NaN-aware weighted mean: a row missing one variable (e.g. a program-scoped
        # indicator like acred_posgrado_ajustado that's only defined for some IES×APDA cells)
        # still gets a score from whichever variables in the dimension it does have,
        # instead of the whole dimension collapsing to NaN.
        var_names = list(vars_in_dim.keys())
        weights   = np.array([vars_in_dim[v] for v in var_names])
        vals      = np.column_stack([norm_cols[v].to_numpy(dtype=float) for v in var_names])
        present   = ~np.isnan(vals)
        w_sum     = present @ weights
        weighted  = np.where(present, vals, 0.0) @ weights
        with np.errstate(invalid='ignore', divide='ignore'):
            dim_scores[dim_name] = np.where(w_sum > 0, weighted / w_sum, np.nan)
        valid_dims.append(dim_name)
        if verbose:
            print(f"  {dim_name:35s}  {len(vars_in_dim)} vars  "
                  f"mean={dim_scores[dim_name].mean():.1f}  std={dim_scores[dim_name].std():.1f}")

    return dim_scores, valid_dims


def _build_axis_flat(axis_cfg, dim_scores, valid_dims, verbose=True):
    """Body of `build_axis` for a single flat {dim: pct} config. Factored out
    so `build_axis` can apply a different flat config per APDA group via
    `axes_by_apda` without duplicating the weighted-mean logic."""
    present = {d: w for d, w in axis_cfg.items() if d in valid_dims}
    missing = [d for d in axis_cfg if d not in valid_dims]
    if missing and verbose:
        print(f"  Warning: dimension(s) not found, skipped: {missing}")
    if not present:
        return None, ""
    total_pct = sum(present.values())
    if abs(total_pct - 100) > 0.5 and verbose:
        print(f"  Warning: axis weights sum to {total_pct:.1f} % (expected 100)")

    dim_names = list(present.keys())
    weights   = np.array([present[d] for d in dim_names])
    vals      = np.column_stack([dim_scores[d].to_numpy(dtype=float) for d in dim_names])
    are_present = ~np.isnan(vals)
    w_sum     = are_present @ weights
    weighted  = np.where(are_present, vals, 0.0) @ weights
    with np.errstate(invalid='ignore', divide='ignore'):
        scores = np.where(w_sum > 0, weighted / w_sum, np.nan)
    label  = "  +  ".join(f"{d} ({w:.0f} %)" for d, w in present.items())
    return pd.Series(scores, index=dim_scores.index), label


def build_axis(axis_cfg, dim_scores, valid_dims, verbose=True,
                apda_col=None, axes_by_apda=None):
    """Composite axis = NaN-aware weighted mean of dimension scores (0-100 scale).

    Shared by Vector/01_vectors.py, Dashboard/build_dashboard.py and
    Sensitivity/ — previously duplicated verbatim as `_build_axis` /
    `build_axis` in the first two; this is now the single source of truth,
    called through `run_full_pipeline` below or directly.

    A row missing one dimension (e.g. an APDA with no posgrado offerings at
    all, so 'Investigación ANID (APDA)' is NaN there) still gets an axis
    value from whichever dimensions it does have, instead of the whole axis
    going NaN.

    `apda_col`/`axes_by_apda`: optional per-APDA weight override. When
    `axes_by_apda` (a dict {apda_name: {dim: pct}}) is given, each group of
    `dim_scores` sharing an `apda_col` value uses its own entry if present,
    falling back to `axis_cfg` otherwise — this is config.yaml's
    `axes_by_apda`. Leaving `axes_by_apda` as None (the default) reproduces
    the original single-flat-config behavior exactly.

    Retorna (scores, label): scores es un pd.Series alineado con el índice de
    dim_scores (o None si ninguna dimensión de axis_cfg es válida).
    """
    if not axes_by_apda:
        return _build_axis_flat(axis_cfg, dim_scores, valid_dims, verbose=verbose)

    parts = []
    for apda_val, group in dim_scores.groupby(apda_col, sort=False):
        cfg = axes_by_apda.get(apda_val, axis_cfg)
        scores, _ = _build_axis_flat(cfg, group, valid_dims, verbose=verbose)
        if scores is not None:
            parts.append(scores)
    if not parts:
        return None, ""
    combined = pd.concat(parts).reindex(dim_scores.index)
    label = "  +  ".join(f"{d} ({w:.0f} %)" for d, w in axis_cfg.items() if d in valid_dims)
    return combined, label


def run_full_pipeline(agg, dimensions, axes, id_cols, target_col=None,
                       norm_method='zscore_within_apda', apda_col='apda',
                       direction=None, filter_col=None, filter_min=None,
                       verbose=True, axes_by_apda=None):
    """End-to-end vector-map pipeline: variable normalization -> dimension
    scores -> optional accreditation-years filter -> composite x/y axes ->
    within-APDA percentile rank.

    This is the single call every Sensitivity/ perturbation scenario makes
    with a modified `dimensions`/`axes`/`norm_method`/`direction`/`filter_min`
    — it reproduces exactly what Vector/01_vectors.py's manual scatter and
    Dashboard/build_dashboard.py's 'Mapa vectorial' plot do, so a baseline
    call (unperturbed config, filter_col=target_col, filter_min=ACRED_MIN_FILTER)
    must reproduce their output bit-for-bit. `build_axis` is row-independent
    (a plain weighted mean per row), so filtering right after dim_scores —
    rather than right before the percentile rank, as Vector/01_vectors.py
    does — changes nothing numerically as long as it happens before the
    percentile rank itself, which does depend on which rows are in the group.

    `axes` is {"x": {dim: pct, ...}, "y": {dim: pct, ...}} (i.e. cfg["axes"]).
    `filter_col`/`filter_min`: e.g. target_col / cfg["acred_min_filter"] — if
    both given, keeps only rows with filter_col >= filter_min before ranking.
    `axes_by_apda`: optional {apda_name: {"x": {...}, "y": {...}}} override
    (i.e. cfg["axes_by_apda"]), forwarded to `build_axis` for both axes.
    Defaults to None, which reproduces the original single-axes behavior
    exactly — every existing Sensitivity/ caller leaves this unset.

    Retorna (out, valid_dims, x_label, y_label):
        out — (filtered) dim_scores plus _x_raw/_y_raw (composite axis scores,
              pre-rank) and _x_pct/_y_pct (percentile rank within apda_col,
              0-100 — the coordinates actually plotted on the vector map)
    """
    dim_scores, valid_dims = compute_dimension_scores(
        agg, dimensions, id_cols, target_col=target_col, norm_method=norm_method,
        apda_col=apda_col, direction=direction, verbose=verbose,
    )

    if filter_col is not None and filter_min is not None and filter_col in dim_scores.columns:
        _before = len(dim_scores)
        dim_scores = dim_scores[
            pd.to_numeric(dim_scores[filter_col], errors='coerce') >= filter_min
        ].reset_index(drop=True)
        if verbose:
            print(f"\nFilter >={filter_min} {filter_col}: {_before} -> {len(dim_scores)} rows")

    x_by_apda = {a: cfg['x'] for a, cfg in axes_by_apda.items() if 'x' in cfg} if axes_by_apda else None
    y_by_apda = {a: cfg['y'] for a, cfg in axes_by_apda.items() if 'y' in cfg} if axes_by_apda else None
    x_vals, x_label = build_axis(axes['x'], dim_scores, valid_dims, verbose=verbose,
                                  apda_col=apda_col, axes_by_apda=x_by_apda)
    y_vals, y_label = build_axis(axes['y'], dim_scores, valid_dims, verbose=verbose,
                                  apda_col=apda_col, axes_by_apda=y_by_apda)

    out = dim_scores.copy()
    out['_x_raw'] = x_vals.to_numpy(dtype=float) if x_vals is not None else np.nan
    out['_y_raw'] = y_vals.to_numpy(dtype=float) if y_vals is not None else np.nan
    out['_x_pct'] = out.groupby(apda_col)['_x_raw'].rank(pct=True) * 100.0
    out['_y_pct'] = out.groupby(apda_col)['_y_raw'].rank(pct=True) * 100.0
    return out, valid_dims, x_label, y_label


def _weighted_mean_nan_aware(group, col, weight_col):
    """Media ponderada por weight_col, ignorando NaN en col. Si la suma de
    pesos de las filas con valor no-NaN es 0 (o el peso no existe), cae a
    media simple sobre las filas no-NaN."""
    vals = group[col].to_numpy(dtype=float)
    present = ~np.isnan(vals)
    if not present.any():
        return np.nan
    w = pd.to_numeric(group[weight_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    w_sum = w[present].sum()
    if w_sum > 0:
        return float(np.dot(vals[present], w[present]) / w_sum)
    return float(vals[present].mean())


def aggregate_ies_apda_rich(u, feat_all):
    """Colapsa a nivel IES×APDA con media + desviación estándar intra-celda.

    Para cada feature f en feat_all:
        - anid_*/scimago_*/scival_* (FIRST_PREFIXES): 'first' únicamente (constantes).
        - empleabilidad_*/continuidad_estudios (WEIGHTED_FEATS_PREFIXES): media
          ponderada por WEIGHT_COL (NaN-aware, ver _weighted_mean_nan_aware) →
          columna 'f'  y  std simple (no ponderada) → columna 'f_std'.
        - Variables en SUM_FEATS: suma NaN-aware (min_count=1) → columna 'f'
          y  std → columna 'f_std'. Son headcounts institucionales (ej. 'total
          matrícula primer año'); promediarlas las deflactaría por n_programs.
        - Resto: media simple → columna 'f'  y  std → columna 'f_std'.

    Las columnas f_std de variables institution-constant tendrán std≈0 y serán
    eliminadas naturalmente por el filtro de varianza en los scripts descendientes.

    Retorna agg_df con columnas: ies_norm, apda, TARGET, n_programs, f, f_std…
    """
    first_feats    = [c for c in feat_all if any(c.startswith(p) for p in FIRST_PREFIXES)]
    weighted_feats = [c for c in feat_all
                       if c not in first_feats
                       and any(c.startswith(p) for p in WEIGHTED_FEATS_PREFIXES)]
    sum_feats      = [c for c in feat_all
                       if c not in first_feats and c not in weighted_feats and c in SUM_FEATS]
    mean_feats     = [c for c in feat_all
                       if c not in first_feats and c not in weighted_feats and c not in sum_feats]

    # Safety assert: FIRST columns deben ser constantes dentro de la celda
    for col in first_feats:
        if col not in u.columns:
            continue
        max_distinct = u.groupby(['ies_norm', 'apda'])[col].nunique(dropna=True).max()
        if max_distinct > 1:
            raise ValueError(
                f"aggregate_ies_apda_rich: columna '{col}' mapeada a 'first' pero tiene "
                f"{max_distinct} valores distintos en alguna celda IES×APDA."
            )

    agg_mean = u.groupby(['ies_norm', 'apda'], as_index=False).agg(
        {**{c: 'mean' for c in mean_feats},
         **{c: 'first' for c in first_feats},
         **{c: (lambda s: s.sum(min_count=1)) for c in sum_feats},
         TARGET: 'first'}
    )

    if weighted_feats:
        groups = u.groupby(['ies_norm', 'apda'])
        agg_weighted = groups.apply(
            lambda g: pd.Series({c: _weighted_mean_nan_aware(g, c, WEIGHT_COL)
                                  for c in weighted_feats}),
            include_groups=False,
        ).reset_index()
        agg_mean = agg_mean.merge(agg_weighted, on=['ies_norm', 'apda'], how='left')

    std_feats = mean_feats + weighted_feats + sum_feats
    agg_std = u.groupby(['ies_norm', 'apda'], as_index=False)[std_feats].std(ddof=0)
    agg_std.columns = ['ies_norm', 'apda'] + [f + '_std' for f in std_feats]

    agg_df = agg_mean.merge(agg_std, on=['ies_norm', 'apda'])
    agg_df['n_programs'] = u.groupby(['ies_norm', 'apda']).size().values

    return agg_df


def _parity_score(pct_mujeres, band_low=0.40, band_high=0.60):
    """Trapezoidal membership on a 0-1 women's-share input (same 0-1 convention
    as pct_matricula_hombres/mujeres). 100 when pct_mujeres in [band_low, band_high],
    tapering linearly to 0 at pct_mujeres = 0.0 or 1.0. NaN in -> NaN out.
    """
    p = pd.Series(pct_mujeres, dtype=float)
    out = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    pv = p[ok]
    s = pd.Series(np.nan, index=pv.index)
    in_band = (pv >= band_low) & (pv <= band_high)
    below   = pv < band_low
    above   = pv > band_high
    s[in_band] = 100.0
    s[below]   = (pv[below] / band_low) * 100.0
    s[above]   = ((1.0 - pv[above]) / (1.0 - band_high)) * 100.0
    out[ok] = s
    return out.clip(lower=0, upper=100)


def add_parity_scores(u, agg_df, id_cols=('ies_norm', 'apda')):
    """Adds paridad_matricula, paridad_matricula_primer_año and
    paridad_jce to agg_df.

    Uses TRUE per-cell area weighting: pct_mujeres = sum(mujeres across all
    programs in the IES×APDA cell) / (sum(mujeres) + sum(hombres)) — each
    program's contribution weighted by its own headcount, not an unweighted
    mean of each program's ratio. Must be called with the program-level
    universe `u` right after aggregate_ies_apda_rich(u, feat_all).

    Denominator intentionally excludes the "no binarios o indefinidos" bucket
    present in all three source families, matching the requested formula.
    """
    id_cols = list(id_cols)
    sums = u.groupby(id_cols).agg(
        _mat_h_sum=('total matrícula hombres', lambda s: s.sum(min_count=1)),
        _mat_m_sum=('total matrícula mujeres', lambda s: s.sum(min_count=1)),
        _mat1_h_sum=('total matrícula hombres primer año', lambda s: s.sum(min_count=1)),
        _mat1_m_sum=('total matrícula mujeres primer año', lambda s: s.sum(min_count=1)),
        _jce_h_sum=('n_de_jce_por_institucion_total_hombres', lambda s: s.sum(min_count=1)),
        _jce_m_sum=('n_de_jce_por_institucion_total_mujeres', lambda s: s.sum(min_count=1)),
    ).reset_index()

    agg_df = agg_df.merge(sums, on=id_cols, how='left')

    pct_mat_mujeres = (
        agg_df['_mat_m_sum'] / (agg_df['_mat_m_sum'] + agg_df['_mat_h_sum'])
    ).replace([np.inf, -np.inf], np.nan)
    pct_mat1_mujeres = (
        agg_df['_mat1_m_sum'] / (agg_df['_mat1_m_sum'] + agg_df['_mat1_h_sum'])
    ).replace([np.inf, -np.inf], np.nan)
    pct_jce_mujeres = (
        agg_df['_jce_m_sum'] / (agg_df['_jce_m_sum'] + agg_df['_jce_h_sum'])
    ).replace([np.inf, -np.inf], np.nan)

    agg_df['paridad_matricula'] = _parity_score(pct_mat_mujeres)
    agg_df['paridad_matricula_primer_año'] = _parity_score(pct_mat1_mujeres)
    agg_df['paridad_jce'] = _parity_score(pct_jce_mujeres)

    return agg_df.drop(columns=[
        '_mat_h_sum', '_mat_m_sum', '_mat1_h_sum', '_mat1_m_sum', '_jce_h_sum', '_jce_m_sum',
    ])


# nivel carrera buckets for acred_posgrado_ajustado (add_acred_scores) and
# prop_matricula_posgrado_apda/prop_matricula_doctorado_apda. 'Posgrado' here
# means Magíster + Especialidad Médica U Odontológica — Doctorado is scored
# separately (as a share of this same population for matrícula, and as its
# own accreditation variable), not folded into the "posgrado" bucket. Shared
# by accreditation and matrícula composition so both use the same "posgrado".
POSGRADO_NIVEL_CARRERA = ['Magister', 'Especialidad Médica U Odontológica']

# vigencia value marking a program still open to new cohorts, as opposed to
# 'Vigente sin estudiantes nuevos'. Used by add_acred_scores: a program
# closed to new students has nothing left to accredit for a prospective one.
VIGENCIA_CON_ESTUDIANTES_NUEVOS = 'Vigente con estudiantes nuevos'

# Spec per acred_* score: which nivel carrera rows feed it, whether it's
# masked to a single APDA, whether it uses Bayesian shrinkage (vs. raw k/n),
# and the names of its k/n trazabilidad columns. See add_acred_scores.
ACRED_SCORE_SPECS = {
    'acred_posgrado_ajustado': dict(
        nivel_carrera=POSGRADO_NIVEL_CARRERA, apda_mask=None, shrink=True,
        n_col='n_posgrado_programs', k_col='k_posgrado_acreditados',
    ),
    'acred_doctorado_ajustado': dict(
        nivel_carrera=['Doctorado'], apda_mask=None, shrink=True,
        n_col='n_doctorado_programs', k_col='k_doctorado_acreditados',
    ),
    'acred_especialidad_medica': dict(
        nivel_carrera=['Especialidad Médica U Odontológica'],
        apda_mask=SALUD_GROUPS, shrink=False,
        n_col='n_especialidad_programs', k_col='k_especialidad_acreditadas',
    ),
}


def _shrink_kn(kn, m_col='_m_apda', C_col='_C_apda', k_col='k', n_col='n'):
    """Bayesian/IMDb-style shrinkage of a per-cell k/n ratio toward the
    APDA-wide accreditation rate m, weighted by pseudo-count C (average n
    across cells in that APDA that do have offerings). Cells with
    about-average evidence stay close to their own ratio; cells with little
    evidence get pulled toward the APDA prior.

        shrunk = (k + C·m) / (n + C)

    Both m and C must be computed per APDA (not pooled globally) — otherwise
    a cell's shrunk score would partly reflect the evidence level of OTHER
    APDAs it doesn't even participate in. Caller is responsible for only
    reading this for n > 0 rows (m_col/C_col are undefined where the APDA has
    no cell with n > 0, which cannot happen for a cell that itself has n > 0).
    """
    return (kn[k_col] + kn[C_col] * kn[m_col]) / (kn[n_col] + kn[C_col])


def add_acred_scores(u, agg_df, id_cols=('ies_norm', 'apda')):
    """Adds acred_posgrado_ajustado, acred_doctorado_ajustado and
    acred_especialidad_medica to agg_df, plus each one's k_*/n_* trazabilidad
    columns (see ACRED_SCORE_SPECS).

    Shared pipeline for the three variables:
      1. Restrict to vigencia == 'Vigente con estudiantes nuevos'
         (VIGENCIA_CON_ESTUDIANTES_NUEVOS) — a program closed to new cohorts
         has nothing left to accredit for a prospective student.
      2. Deduplicate to one row per (id_cols, cod_carrera): master_dataset's
         grain is (inst, sede, carrera, jornada, versión), so the same SIES
         program repeats across sedes/jornadas/versiones (fan-out ~1.1x in
         posgrado/doctorado). acred_bin is collapsed with max — if any
         version of the program is accredited, the program counts as
         accredited (5 of 1,679 posgrado programs have mixed status across
         versions).

    Three explicit states per cell, computed with named, mutually exclusive
    masks (never inferred from assignment order):
      - sin_oferta (n == 0): no vigente-con-nuevos program of that level in
        that IES×APDA cell -> NaN. Downstream NaN-aware weighted means
        (compute_dimension_scores, _build_axis) score the cell from whichever
        other variables it does have, instead of this pulling it toward 0.
      - cero_acreditados (n > 0, k == 0): offers the level, none accredited
        -> forced to 0.0. Shrinkage exists to stop a small n from overstating
        a GOOD ratio (3/3 shouldn't beat 5/15); it isn't meant to soften a
        confirmed zero.
      - con_acreditados (n > 0, k > 0): acred_posgrado_ajustado/
        acred_doctorado_ajustado use _shrink_kn (Bayesian shrinkage toward
        the APDA rate). acred_especialidad_medica uses the raw ratio k/n
        instead — n averages ~24 programs there, well past the small-
        denominator regime shrinkage was built for — but keeps the same
        three-state split.

    acred_especialidad_medica is additionally masked to the "Salud y
    bienestar" APDA: in the current data it is already 100% scoped there by
    nivel carrera (verified empirically), so this is a no-op today, but it
    documents the intent and guards against a future APDA remap. The mask is
    applied by zeroing k/n for out-of-APDA cells *before* computing the three
    states, so a masked-out cell reads as sin_oferta (NaN) exactly like a
    real absence of offering, keeping the k_col/n_col trazabilidad columns
    consistent with the exported value.

    k_*_acreditados/n_*_programs are exported as integers, always 0 (never
    NaN) when there's no offering — they make the NaN/0 distinction legible
    directly from the data instead of requiring the reader to infer it from a
    NaN. Invariant, asserted below: var.isna() <=> n_col == 0.

    Must be called with the program-level universe `u` (same as
    aggregate_ies_apda_rich) and an agg_df whose id_cols already cover every
    IES×APDA cell in u (true right after aggregate_ies_apda_rich).
    """
    id_cols = list(id_cols)
    cells = u[id_cols].drop_duplicates()
    base = u[u['vigencia'] == VIGENCIA_CON_ESTUDIANTES_NUEVOS]

    out = agg_df
    for var_name, spec in ACRED_SCORE_SPECS.items():
        sub = base[base['nivel carrera'].isin(spec['nivel_carrera'])]
        prog = sub.groupby(id_cols + ['cod_carrera'], as_index=False)['acred_bin'].max()

        kn = prog.groupby(id_cols)['acred_bin'].agg(k='sum', n='count').reset_index()
        kn = cells.merge(kn, on=id_cols, how='left')
        kn[['k', 'n']] = kn[['k', 'n']].fillna(0.0)

        if spec['apda_mask'] is not None:
            outside = ~kn['apda'].isin(spec['apda_mask'])
            kn.loc[outside, ['k', 'n']] = 0.0

        sin_oferta = kn['n'] == 0
        cero_acreditados = (kn['n'] > 0) & (kn['k'] == 0)
        con_acreditados = (kn['n'] > 0) & (kn['k'] > 0)

        if spec['shrink']:
            m_by_apda = prog.groupby('apda')['acred_bin'].mean().rename('_m_apda')
            kn = kn.merge(m_by_apda, on='apda', how='left')
            C_by_apda = kn.loc[kn['n'] > 0].groupby('apda')['n'].mean().rename('_C_apda')
            kn = kn.merge(C_by_apda, on='apda', how='left')
            ratio = _shrink_kn(kn)
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = kn['k'] / kn['n']

        kn[var_name] = np.select(
            [sin_oferta, cero_acreditados, con_acreditados],
            [np.nan, 0.0, ratio],
            default=np.nan,
        )
        kn[spec['n_col']] = kn['n'].astype(int)
        kn[spec['k_col']] = kn['k'].astype(int)

        assert (kn[var_name].isna() == (kn[spec['n_col']] == 0)).all(), (
            f"{var_name}: NaN debe corresponder exactamente a {spec['n_col']} == 0 "
            "(sin oferta), nunca a cero programas acreditados."
        )

        out = out.merge(
            kn[id_cols + [var_name, spec['n_col'], spec['k_col']]],
            on=id_cols, how='left',
        )

    return out


def add_empleabilidad_ponderada(u, agg_df, id_cols=('ies_norm', 'apda')):
    """Adds empleabilidad_pond_1er_año and empleabilidad_pond_2do_año to agg_df.

    Para cada spec de EMP_POND_SPECS (emp_col, weight_col):
      1. Colapsa u a una fila por (id_cols, EMP_UNIT_KEYS) — quita el fan-out
         sede/jornada/versión del match de empleabilidad (empleabilidad es
         constante dentro de esa unidad; verificado sobre datos reales).
      2. Colapsa otra vez a una fila por (id_cols, EMP_GEN_COL) — dentro de una
         IES, más de un cod_carrera puede compartir carrera genérica; sin este
         paso su peso (weight_col, que ya está a nivel genérica) entraría
         duplicado en la suma.
      3. Por celda: value = Σ(emp·weight) / Σ(weight), sobre las genéricas con
         emp no nulo. Si Σ(weight) es 0 o weight no existe para ninguna,
         cae a media simple de emp (mismo criterio que
         _weighted_mean_nan_aware) para no perder cobertura.

    Debe llamarse con el universo programa-nivel `u` (mismo que
    aggregate_ies_apda_rich).
    """
    id_cols = list(id_cols)
    out = agg_df

    for pond_col, (emp_col, weight_col) in EMP_POND_SPECS.items():
        cols_needed = id_cols + EMP_UNIT_KEYS + [EMP_GEN_COL, emp_col, weight_col]
        sub = u.loc[u[emp_col].notna(), cols_needed].copy()
        sub = sub.drop_duplicates(id_cols + EMP_UNIT_KEYS)
        sub = sub.drop_duplicates(id_cols + [EMP_GEN_COL])

        def _f_n(group, emp_col=emp_col, weight_col=weight_col):
            emp = group[emp_col].to_numpy(dtype=float)
            w = pd.to_numeric(group[weight_col], errors='coerce').to_numpy(dtype=float)
            present_w = ~np.isnan(w)
            w_sum = w[present_w].sum() if present_w.any() else 0.0
            if w_sum > 0:
                return float(np.dot(emp[present_w], w[present_w]) / w_sum)
            return float(emp.mean())

        pond = sub.groupby(id_cols).apply(_f_n, include_groups=False)
        pond = pond.rename(pond_col).reset_index()
        out = out.merge(pond, on=id_cols, how='left')

    return out


def add_prop_matricula_posgrado_apda(u, agg_df, id_cols=('ies_norm', 'apda')):
    """Adds prop_matricula_posgrado_apda, prop_matricula_doctorado_apda and
    prop_matricula_especialidad_apda to agg_df.

    Replaces the retired prop_matricula_posgrado_ies (institution-constant,
    broadcast to every APDA row from Build_master_dataset.py) with a real
    per-IES×APDA composition ratio, following the same true per-cell
    aggregation pattern as add_parity_scores/add_acred_scores. Uses the same
    POSGRADO_NIVEL_CARRERA bucket as add_acred_scores' acred_posgrado_ajustado
    so "posgrado" means the same population in matrícula and accreditation:

        prop_matricula_posgrado_apda      = mat(Magister + Esp. Médica U Odont.) / mat_total
        prop_matricula_doctorado_apda     = mat(Doctorado) / mat(Magister + Esp. Médica U Odont.)
        prop_matricula_especialidad_apda  = mat(Esp. Médica U Odont.) / mat(Magister + Esp. Médica U Odont.)

    prop_matricula_especialidad_apda is additionally masked to the "Salud y
    bienestar" APDA, mirroring add_acred_scores' acred_especialidad_medica:
    Especialidad Médica U Odontológica programs are already scoped there by
    nivel carrera, so the mask is a no-op today, but it documents the intent
    and guards against a future APDA remap.

    All four sums are grouped by id_cols (default IES×APDA cell). A missing
    posgrado/doctorado/especialidad sum after the left-merge means zero
    programs of that kind in the cell (a real zero, not missing data), so
    those three are filled with 0.0 before dividing — mat_total is never
    filled, so a cell truly absent from u still yields NaN throughout.

    Must be called with the program-level universe `u` (same as
    aggregate_ies_apda_rich).
    """
    id_cols = list(id_cols)
    _is_posgrado = u['nivel carrera'].isin(POSGRADO_NIVEL_CARRERA)
    _is_doctorado = u['nivel carrera'] == 'Doctorado'
    _is_especialidad = u['nivel carrera'] == 'Especialidad Médica U Odontológica'

    sums = u.groupby(id_cols)['total matrícula'].agg(
        lambda s: s.sum(min_count=1)
    ).rename('_mat_total_sum').reset_index()

    _posgrado_sums = (
        u.loc[_is_posgrado].groupby(id_cols)['total matrícula']
        .agg(lambda s: s.sum(min_count=1))
        .rename('_mat_posgrado_sum').reset_index()
    )
    _doctorado_sums = (
        u.loc[_is_doctorado].groupby(id_cols)['total matrícula']
        .agg(lambda s: s.sum(min_count=1))
        .rename('_mat_doctorado_sum').reset_index()
    )
    _especialidad_sums = (
        u.loc[_is_especialidad].groupby(id_cols)['total matrícula']
        .agg(lambda s: s.sum(min_count=1))
        .rename('_mat_especialidad_sum').reset_index()
    )

    sums = sums.merge(_posgrado_sums, on=id_cols, how='left')
    sums = sums.merge(_doctorado_sums, on=id_cols, how='left')
    sums = sums.merge(_especialidad_sums, on=id_cols, how='left')
    sums['_mat_posgrado_sum'] = sums['_mat_posgrado_sum'].fillna(0.0)
    sums['_mat_doctorado_sum'] = sums['_mat_doctorado_sum'].fillna(0.0)
    sums['_mat_especialidad_sum'] = sums['_mat_especialidad_sum'].fillna(0.0)

    agg_df = agg_df.merge(sums, on=id_cols, how='left')

    agg_df['prop_matricula_posgrado_apda'] = (
        agg_df['_mat_posgrado_sum'] / agg_df['_mat_total_sum']
    ).replace([np.inf, -np.inf], np.nan)
    agg_df['prop_matricula_doctorado_apda'] = (
        agg_df['_mat_doctorado_sum'] / agg_df['_mat_posgrado_sum']
    ).replace([np.inf, -np.inf], np.nan)
    agg_df['prop_matricula_especialidad_apda'] = (
        agg_df['_mat_especialidad_sum'] / agg_df['_mat_posgrado_sum']
    ).replace([np.inf, -np.inf], np.nan).where(agg_df['apda'].isin(SALUD_GROUPS))

    return agg_df.drop(columns=[
        '_mat_total_sum', '_mat_posgrado_sum', '_mat_doctorado_sum', '_mat_especialidad_sum',
    ])


# pct_tes_particular_pagado_apda: variable de "Costo" — % de estudiantes con
# dependencia de origen (TES) conocida que vienen de un establecimiento
# secundario particular pagado, dentro de cada celda IES×APDA. Grano fuente:
# I/S/C/J/V (institución×sede×carrera×jornada×versión) — a diferencia de
# PUNTAJE_UNIT_KEYS (que colapsa jornada porque el puntaje CNED no varía por
# ella), acá la jornada sí debe mantenerse distinta (a pedido del usuario),
# así que PROGRAM_UNIT_KEYS sólo colapsa versión.
PROGRAM_UNIT_KEYS = ['cod_inst', 'cod_sede', 'cod_carrera', 'cod_jornada']


def add_pct_tes_particular_pagado_apda(u, agg_df, id_cols=('ies_norm', 'apda')):
    """Adds pct_tes_particular_pagado_apda (0-100) to agg_df.

        pct_tes_particular_pagado_apda =
            100 * sum(tes particular pagado) / sum(total tes)

    Two-step sum: first collapse to PROGRAM_UNIT_KEYS (I/S/C/J, ignoring
    versión) so duplicate versiones of the same program/jornada don't
    inflate the numerator/denominator, then sum those program-level totals
    up to id_cols (IES×APDA), matching the grain used by
    add_prop_matricula_posgrado_apda/add_parity_scores/add_acred_scores.

    Must be called with the program-level universe `u` (same as
    aggregate_ies_apda_rich).
    """
    id_cols = list(id_cols)

    prog = u.groupby(PROGRAM_UNIT_KEYS + id_cols).agg(
        _total_tes_prog=('total tes', lambda s: s.sum(min_count=1)),
        _tes_part_pagado_prog=('tes particular pagado', lambda s: s.sum(min_count=1)),
    ).reset_index()

    sums = prog.groupby(id_cols).agg(
        _total_tes_sum=('_total_tes_prog', lambda s: s.sum(min_count=1)),
        _tes_part_pagado_sum=('_tes_part_pagado_prog', lambda s: s.sum(min_count=1)),
    ).reset_index()

    agg_df = agg_df.merge(sums, on=id_cols, how='left')

    agg_df['pct_tes_particular_pagado_apda'] = (
        100 * agg_df['_tes_part_pagado_sum'] / agg_df['_total_tes_sum']
    ).replace([np.inf, -np.inf], np.nan)

    return agg_df.drop(columns=['_total_tes_sum', '_tes_part_pagado_sum'])


# Variables CNED de "Selectividad": promedio_puntaje/puntaje_corte
# reemplazan a promedio_puntaje/puntaje_corte (retirados de FORCE_INCLUDE).
# Grano fuente: institución×sede×carrera×jornada×versión (I/S/C/J/V), pero ya
# matcheado a I/S/C en Build_master_dataset.py (jornada/versión de una misma
# carrera comparten el mismo valor CNED) — sin deduplicar por I/S/C antes de
# promediar, esas jornadas/versiones repetidas inflarían el peso de la
# carrera dentro de la celda IES×APDA. minimo_puntaje queda retirado (no se
# calcula su versión _apda).
PUNTAJE_VARS = ['promedio_puntaje', 'puntaje_corte']
PUNTAJE_UNIT_KEYS = ['cod_inst', 'cod_sede', 'cod_carrera']


def add_puntaje_seleccion(u, agg_df, id_cols=('ies_norm', 'apda')):
    """Adds promedio_puntaje_apda, puntaje_corte_apda and the shared trace
    column n_puntaje_programs to agg_df.

    Por variable en PUNTAJE_VARS:
      1. Restringe a filas con valor no-NaN.
      2. Deduplica a una fila por (id_cols, PUNTAJE_UNIT_KEYS) — el match
         ampliado a I/S/C en Build_master_dataset.py replica un mismo valor
         CNED sobre varias filas de jornada/versión del mismo programa; sin
         este paso esas copias contarían más de una vez en la celda.
      3. Media simple (sin ponderar por matrícula ni ningún otro tamaño) por
         celda id_cols — composición de programas (cuántas jornadas/sedes
         ofrece una carrera) no debe influir en el resultado.

    n_puntaje_programs cuenta los programas (I/S/C) distintos que aportan a
    la celda; promedio_puntaje y puntaje_corte son siempre co-nulos en
    cned_programa.csv (verificado: 2025/2025 filas), así que una sola columna
    de conteo sirve para ambas variables. Celdas sin ningún programa CNED
    quedan en NaN (n_puntaje_programs == 0), mismo criterio "sin oferta" que
    add_acred_scores.

    Debe llamarse con el universo programa-nivel `u` (mismo que
    aggregate_ies_apda_rich).
    """
    id_cols = list(id_cols)
    out = agg_df
    n_prog = None

    for var in PUNTAJE_VARS:
        sub = u.loc[u[var].notna(), id_cols + PUNTAJE_UNIT_KEYS + [var]]
        sub = sub.drop_duplicates(id_cols + PUNTAJE_UNIT_KEYS)

        agg_var = sub.groupby(id_cols, as_index=False)[var].mean()
        agg_var = agg_var.rename(columns={var: f'{var}_apda'})
        out = out.merge(agg_var, on=id_cols, how='left')

        count = sub.groupby(id_cols).size().rename('n_puntaje_programs').reset_index()
        n_prog = count if n_prog is None else n_prog

    cells = u[id_cols].drop_duplicates()
    n_prog = cells.merge(n_prog, on=id_cols, how='left')
    n_prog['n_puntaje_programs'] = n_prog['n_puntaje_programs'].fillna(0).astype(int)
    out = out.merge(n_prog, on=id_cols, how='left')

    return out
