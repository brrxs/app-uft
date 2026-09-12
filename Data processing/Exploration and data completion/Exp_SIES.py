# %% Set-up

import pandas as pd
import os
import unicodedata


apda_grupos = {
    "Salud y bienestar":                      ["Salud y Bienestar"],
    "Artes, humanidades y cultura":           ["Artes y Humanidades"],
    "Ed., Sociedad y desarrollo humano":      ["Administración de Empresas y Derecho",
                                               "Ciencias Sociales, Periodismo e Información",
                                               "Educación",
                                               "Servicios"],
    "Ciencias, tecnología y calidad de vida": ["Ciencias naturales, matemáticas y estadística",
                                               "Ingeniería, Industria y Construcción",
                                               "Tecnología de la Información y la Comunicación (TIC)"],}

area_to_group = {area: group
    for group, areas in apda_grupos.items()
    for area in areas}


def ins(x):
    for col in x:
        print(col)


def normalizar_nombre_columna(col):
    col = str(col).strip().lower()
    col = unicodedata.normalize("NFKD", col)
    col = ''.join(c for c in col if not unicodedata.combining(c))
    col = col.replace(" ", "_")
    col = col.replace("–", "_").replace("—", "_")
    col = col.replace("°", "")
    col = col.replace(".", "_")
    col = col.replace("%", "porc")
    return col

periodo = [2025]

# %% MAtrícula - SIES

mat_SIES= pd.read_csv('datos/SIES/Matricula_2007_2025_WEB_15_07_2025.csv',
                      encoding= 'latin-1',
                      delimiter=';')


mat_SIES["AÑO"] = mat_SIES["AÑO"].str.extract(r"(\d{4})").astype(int)
print(mat_SIES["AÑO"].unique())

mask = (
    (mat_SIES["CLASIFICACIÓN INSTITUCIÓN NIVEL 1"] == "Universidades") &
    # (mat_SIES["NIVEL GLOBAL"] != "Postítulo") &
    (mat_SIES["AÑO"].isin(periodo)))
            
mat_SIES = mat_SIES[mask]


mat_SIES['APDA'] = mat_SIES['CINE-F 2013 ÁREA'].map(area_to_group)
mat_SIES['APDA'] = mat_SIES['APDA'].fillna('No aplica')

mat_SIES['Fuente'] ='Matricula' 
mat_SIES['Organismo'] ='SIES' 

# %% Oferta académica - SIES


of_SIES = pd.read_csv('datos/SIES/Oferta_Academica_2010_al_2025_SIES_02_06_2025_WEB_E.csv',
                      encoding='latin-1',
                      delimiter= ';')


of_SIES["Año"] = of_SIES["Año"].str.extract(r"(\d{4})").astype(int)

of_SIES['APDA'] = of_SIES['Cine-F 13 Área'].map(area_to_group)
of_SIES['APDA'] = of_SIES['APDA'].fillna('No aplica')

of_SIES['Fuente'] ='Oferta' 
of_SIES['Organismo'] ='SIES' 

print(of_SIES['Nivel Global'].unique())

mask1 = (
    (of_SIES["Tipo Institución 1"] == "Universidades") &
    # (of_SIES["Nivel Global"] != "Postítulo") &
    (of_SIES["Año"].isin(periodo)))

of_SIES= of_SIES[mask1]

of_SIES['Nivel Global'] = of_SIES['Nivel Global'].replace('Postgrado', 'Posgrado')


# %% Merge con supuestos

# =============================================================================
# # Este merge toma los siguientes supuestos:
#     # 1. Matrícula tiene la información definitiva
#     # 2. oferta tiene programas que no están en matrícula
#     # 3. El merge conservador (5 keys) es correcto 
# =============================================================================


mat_SIES.groupby('NIVEL GLOBAL')['CÓDIGO CARRERA'].count()
of_SIES.groupby('Nivel Global')['Código Carrera'].count()

mat_SIES.columns = mat_SIES.columns.str.lower()
of_SIES.columns = of_SIES.columns.str.lower()

ins(mat_SIES)
ins(of_SIES)

# Variables presentes en ambos (conceptualmente equivalentes)
col_map = {
    # of_col                        : mat_col
    'área del conocimiento'         : 'área del conocimiento',
    'nombre carrera'                : 'nombre carrera',
    'nombre sede'                   : 'nombre sede',
    'modalidad'                     : 'modalidad',
    'jornada'                       : 'jornada',
    'nivel global'                  : 'nivel global',
    'apda'                          : 'apda',
    'fuente'                        : 'fuente',
    'organismo'                     : 'organismo',
}

# Variables exclusivas de of a traer a mat
cols_only_of = [
    'vigencia',
    'grado académico',
    'nivel carrera',
    'tipo carrera',
    'acreditación carrera o programa',
    'duración estudios',
    'duración titulación',
    'duración total',
    'vacantes semestre uno',
    'vacantes semestre dos',
    'arancel anual',
    'matrícula anual',
    'nombre título',
    'año inicio',
]


# Renombrar columnas de of que se solapan con mat
of_rename = {of_col: mat_col for of_col, mat_col in col_map.items()}
of_for_merge = of_SIES.rename(columns=of_rename)

# Seleccionar solo las columnas que vamos a traer desde of
keys_5 = ['cod_inst','cod_sede','cod_carrera','cod_jornada','cod_version']

of_SIES[keys_5] = (
    of_SIES['código único']
    .str.extract(r'I(\d+)S(\d+)C(\d+)J(\d+)V(\d+)')
    .astype('Int64'))

mat_SIES[keys_5] = (
    mat_SIES['código carrera']
    .str.extract(r'I(\d+)S(\d+)C(\d+)J(\d+)V(\d+)')
    .astype('Int64'))


# Traer todas las columnas de mat que no están ya en of (of tiene prioridad)
_shared_with_of = set(of_SIES.columns) - set(keys_5)
mat_slim = mat_SIES[[c for c in mat_SIES.columns if c not in _shared_with_of]]

# Merge: of es el recipiente
merged_final = of_SIES.merge(mat_slim, on=keys_5, how='left', indicator=True)
merged_final['flag_matricula'] = (merged_final['_merge'] == 'both').astype(int)
merged_final = merged_final.drop(columns='_merge')

# Diagnóstico
print(f"mat filas original:  {len(mat_SIES)}")
print(f"merged filas:        {len(merged_final)}")
print(f"fan_out:             {len(merged_final) - len(mat_SIES)}")
print(f"match rate:          {merged_final['vigencia'].notna().mean():.1%}")
print(f"sin match:           {merged_final['vigencia'].isna().sum()}")



# %% no-match


merged_indicator = of_SIES.merge(mat_SIES[keys_5].drop_duplicates(), 
                                  on=keys_5, how='left', indicator=True)

in_of_not_mat = merged_indicator[merged_indicator['_merge'] == 'left_only'].drop(columns='_merge')

print(f"En of pero no en mat: {len(in_of_not_mat)}")

print("\n=== Nivel Global ===")
print(in_of_not_mat['nivel global'].value_counts())

print("\n=== Vigencia ===")
print(in_of_not_mat['vigencia'].value_counts())

print("\n=== Nivel Global x Vigencia ===")
print(in_of_not_mat.groupby(['nivel global','vigencia']).size().reset_index(name='n').to_string())

# =============================================================================
# # quiza´la vigencia es lo que me está sacando programas
# # Vigente sin estudiantes nuevos    3102
# # Vigente con estudiantes nuevos     349 --> esto pueden ser programas que parten el 2do semestre
# #       ej. doctorado educación UFT partió el 2do semestre 
# 
# =============================================================================

# %%% Eval. progrsiva de keys
stage_keys = [
    ['cod_inst', 'cod_carrera'],
    ['cod_inst', 'cod_carrera', 'cod_jornada'],
    ['cod_inst', 'cod_carrera', 'cod_jornada', 'cod_version'],
]

print("stage | en ambos | solo merged_final | solo in_of_not_mat")
print("-" * 65)

for keys in stage_keys:
    mf_keys  = merged_final[keys].drop_duplicates()
    nm_keys  = in_of_not_mat[keys].drop_duplicates()
    
    m = mf_keys.merge(nm_keys, on=keys, how='outer', indicator=True)
    
    ambos    = (m['_merge'] == 'both').sum()
    solo_mf  = (m['_merge'] == 'left_only').sum()
    solo_nm  = (m['_merge'] == 'right_only').sum()
    
    label = ' + '.join([k.replace('cod_','') for k in keys])
    print(f"{label:<40} | {ambos:>8} | {solo_mf:>17} | {solo_nm:>18}")


# %%% ins merge_f

ins(merged_final)
print(merged_final['acreditación institucional'].unique())

# %% Datos agregados nivel IES - SIES

bsc_inst=pd.read_excel("datos/SIES/Buscador_Instituciones_2025_2026_SIES-vf.xlsx",
                       header=[1])
print("\n".join(bsc_inst.columns.tolist()))

exec(open("Utils/utils.py", encoding="utf-8").read())

# Acá quiero usar esta base como la base para futuros nombres y matchs

# Primer filtro
ins(bsc_inst,"Años acreditación (30 de octubre de 2025)")
ins(bsc_inst,"Tipo de institución")


mask1 = (bsc_inst["Tipo de institución"] =='Universidades')
bsc_inst=bsc_inst[mask1]


ins(bsc_inst,"Nombre institución")

bsc_inst["ies_norm"]  = bsc_inst["Nombre institución"].apply(normalizar_nombre_ies)
bsc_inst["ies_abrev"] = bsc_inst["ies_norm"].map(abreviaciones_ies).fillna(bsc_inst["ies_norm"])


ins(bsc_inst,"ies_norm")

# merge con el resto 
print("\n".join(merged_final.columns.tolist()))

_nombre_ies = merged_final["nombre institución"].fillna(merged_final["nombre ies"])
merged_final["ies_norm"]  = _nombre_ies.apply(normalizar_nombre_ies)
merged_final["ies_abrev"] = merged_final["ies_norm"].map(abreviaciones_ies).fillna(bsc_inst["ies_norm"])

merged_final['ies_norm'] = merged_final['ies_norm'].replace({
    'universidad de santiago de chile (* carrera en convenio u. iberoamericana)': 'universidad de santiago de chile',
    'universidad de artes, ciencias y comunicacion - uniacc': 'universidad de artes, ciencias y comunicacion uniacc',
})

ins(merged_final,"ies_norm")

# Seleccionar columnas relevantes de bsc_inst para traer a merged_final
cols_bsc = [
    'ies_norm',
    'ies_abrev',
    'Años acreditación (30 de octubre de 2025)',
    ]

bsc_slim = bsc_inst[cols_bsc].drop_duplicates(subset='ies_norm')

# Merge
merged_final = merged_final.merge(bsc_slim, on='ies_norm', how='left', suffixes=('', '_bsc'))

# Diagnóstico
print(f"merged_final filas:     {len(merged_final)}")
print(f"con match en bsc_inst:  {merged_final['Años acreditación (30 de octubre de 2025)'].notna().sum()}")
print(f"sin match en bsc_inst:  {merged_final['Años acreditación (30 de octubre de 2025)'].isna().sum()}")

print("\n=== IES sin match ===")
sin_match = merged_final[merged_final['Años acreditación (30 de octubre de 2025)'].isna()]['ies_norm'].unique()
print(sin_match)


# %% PAC Académicos JCE - SIES

pac_raw = pd.read_excel(
    'datos/SIES/PAC_web_2008_2025_SIES_EE.xlsx',
    sheet_name=1,  # BD_Académicos_JCE
    header=None,
)

# Flatten 3-row header: ffill row 0 for merged cells; ffill row 1 only within row-0 groups
row0 = pac_raw.iloc[0].copy()
row1 = pac_raw.iloc[1].copy()
row2 = pac_raw.iloc[2].copy()

row0_ff = row0.ffill()

row1_ff = row1.copy()
_cur_r1 = None
for j in range(len(row1)):
    if pd.notna(row0.iloc[j]):
        _cur_r1 = None
    if pd.notna(row1.iloc[j]):
        _cur_r1 = row1.iloc[j]
    row1_ff.iloc[j] = _cur_r1

combined_cols = []
for j in range(pac_raw.shape[1]):
    parts = []
    for val in [row0_ff.iloc[j], row1_ff.iloc[j], row2.iloc[j]]:
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s and s != 'nan':
            parts.append(s)
    combined_cols.append(' '.join(parts) if parts else f'col_{j}')

pac = pac_raw.iloc[3:].copy()
pac.columns = combined_cols
pac = pac.reset_index(drop=True)
pac.columns = [normalizar_nombre_columna(c) for c in pac.columns]

# Filter by institution type
pac = pac[pac['tipo_institucion_i'] == 'Universidades'].copy()

# Filter by year (Periodo = "PAC_####")
pac['_año'] = pac['periodo'].str.extract(r'(\d{4})').astype(int)
pac = pac[pac['_año'].isin(periodo)].copy()

# Normalize institution name for merge key
pac['ies_norm'] = pac['nombre_institucion'].apply(normalizar_nombre_ies)
pac['ies_norm'] = pac['ies_norm'].replace({
    'universidad de artes, ciencias y comunicacion - uniacc': 'universidad de artes, ciencias y comunicacion uniacc',
})

# Drop administrative/identifier columns; keep only PAC data cols + join key
_id_cols_pac = {'periodo', 'codigo_institucion', 'nombre_institucion',
                'tipo_institucion_i', 'tipo_institucion_ii', 'tipo_institucion_iii', '_año'}
pac_slim = pac[['ies_norm'] + [c for c in pac.columns if c not in _id_cols_pac | {'ies_norm'}]]
pac_slim = pac_slim.drop_duplicates(subset='ies_norm')

# Merge: broadcast institution-level PAC data to every program row
merged_final = merged_final.merge(pac_slim, on='ies_norm', how='left')

_first_pac = [c for c in pac_slim.columns if c != 'ies_norm'][0]
print(f"Con datos PAC:  {merged_final[_first_pac].notna().sum()}")
print(f"Sin datos PAC:  {merged_final[_first_pac].isna().sum()}")
print("\n=== IES sin match PAC ===")
print(merged_final[merged_final[_first_pac].isna()]['ies_norm'].unique())


# %% PAC Académicos Número - SIES

pac_num_raw = pd.read_excel(
    'datos/SIES/PAC_web_2008_2025_SIES_EE.xlsx',
    sheet_name=0,  # BD_Académicos_Número
    header=None,
)

# Flatten 3-row header: same logic as BD_Académicos_JCE above
row0_num = pac_num_raw.iloc[0].copy()
row1_num = pac_num_raw.iloc[1].copy()
row2_num = pac_num_raw.iloc[2].copy()

row0_num_ff = row0_num.ffill()

row1_num_ff = row1_num.copy()
_cur_r1_num = None
for j in range(len(row1_num)):
    if pd.notna(row0_num.iloc[j]):
        _cur_r1_num = None
    if pd.notna(row1_num.iloc[j]):
        _cur_r1_num = row1_num.iloc[j]
    row1_num_ff.iloc[j] = _cur_r1_num

combined_cols_num = []
for j in range(pac_num_raw.shape[1]):
    parts = []
    for val in [row0_num_ff.iloc[j], row1_num_ff.iloc[j], row2_num.iloc[j]]:
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s and s != 'nan':
            parts.append(s)
    combined_cols_num.append(' '.join(parts) if parts else f'col_{j}')

pac_num = pac_num_raw.iloc[3:].copy()
pac_num.columns = combined_cols_num
pac_num = pac_num.reset_index(drop=True)
pac_num.columns = [normalizar_nombre_columna(c) for c in pac_num.columns]

# Filter by institution type
pac_num = pac_num[pac_num['tipo_institucion_i'] == 'Universidades'].copy()

# Filter by year (Periodo = "PAC_####")
pac_num['_año'] = pac_num['periodo'].str.extract(r'(\d{4})').astype(int)
pac_num = pac_num[pac_num['_año'].isin(periodo)].copy()

# Normalize institution name for merge key
pac_num['ies_norm'] = pac_num['nombre_institucion'].apply(normalizar_nombre_ies)
pac_num['ies_norm'] = pac_num['ies_norm'].replace({
    'universidad de artes, ciencias y comunicacion - uniacc': 'universidad de artes, ciencias y comunicacion uniacc',
})

# "39 y más" contracted-hours bracket, plus the institution's total headcount
# (needed as the ratio denominator for pct_academicos_39 below)
_col_39 = 'n_de_academicos_por_rango_de_horas_contratadas_entre_39_y_mas'
_col_academicos_total = 'n_de_academicos_por_insitucion_total_general'
pac_num_slim = pac_num[['ies_norm', _col_39, _col_academicos_total]].drop_duplicates(subset='ies_norm')

# Merge: broadcast institution-level academics headcount to every program row
merged_final = merged_final.merge(pac_num_slim, on='ies_norm', how='left')

print(f"Con académicos 39+:  {merged_final[_col_39].notna().sum()}")
print(f"Sin académicos 39+:  {merged_final[_col_39].isna().sum()}")
print("\n=== IES sin match académicos 39+ ===")
print(merged_final[merged_final[_col_39].isna()]['ies_norm'].unique())


# %% Indices Matrícula - CNED

# ind_CNED = pd.read_excel('datos/CNED/BaseINDICES-2005-2025.xlsx')

# # print(ind_CNED['Clasificación1'].unique())
# # print(ind_CNED['Pregrado/Posgrado'].unique())
# # print(ind_CNED['Area Conocimiento'].unique())
# # print(ind_CNED['Grado Académico'].unique())

# print("\n".join(ind_CNED.columns.tolist()))

# ind_CNED = ind_CNED[ind_CNED["Tipo Institución"] == "Univ."].copy()

# ins(ind_CNED, "Tipo Institución")
# ins(ind_CNED, "Nombre Institución")

# # Mapa de normalización CNED → nombre canónico
# ind_CNED_fix = {
#     "u. de chile":                                      "universidad de chile",
#     "pontificia u. catolica de chile":                  "pontificia universidad catolica de chile",
#     "u. de concepcion":                                 "universidad de concepcion",
#     "pontificia u. catolica de valparaiso":             "pontificia universidad catolica de valparaiso",
#     "u. tecnica federico santa maria":                  "universidad tecnica federico santa maria",
#     "u. de santiago de chile":                          "universidad de santiago de chile",
#     "u. austral de chile":                              "universidad austral de chile",
#     "u. catolica del norte":                            "universidad catolica del norte",
#     "u. de la serena":                                  "universidad de la serena",
#     "u. de valparaiso":                                 "universidad de valparaiso",
#     "u. de antofagasta":                                "universidad de antofagasta",
#     "u. de la frontera":                                "universidad de la frontera",
#     "u. de magallanes":                                 "universidad de magallanes",
#     "u. de talca":                                      "universidad de talca",
#     "u. diego portales":                                "universidad diego portales",
#     "u. central de chile":                              "universidad central de chile",
#     "u. de tarapaca":                                   "universidad de tarapaca",
#     "u. arturo prat":                                   "universidad arturo prat",
#     "u. de playa ancha de ciencias de la educacion":    "universidad de playa ancha de ciencias de la educacion",
#     "u. finis terrae":                                  "universidad finis terrae",
#     "u. mayor":                                         "universidad mayor",
#     "u. del bio-bio":                                   "universidad del bio-bio",
#     "u. de las americas":                               "universidad de las americas",
#     "u. adolfo ibanez":                                 "universidad adolfo ibanez",
#     "u. andres bello":                                  "universidad andres bello",
#     "u. del desarrollo":                                "universidad del desarrollo",
#     "u. san sebastian":                                 "universidad san sebastian",
#     "u. autonoma de chile":                             "universidad autonoma de chile",
#     "u. de los andes":                                  "universidad de los andes",
#     "u. bernardo o`higgins":                            "universidad bernardo o'higgins",
#     "u. catolica del maule":                            "universidad catolica del maule",
#     "u. catolica de la santisima concepcion":           "universidad catolica de la santisima concepcion",
#     "u. catolica de temuco":                            "universidad catolica de temuco",
#     "u. de los lagos":                                  "universidad de los lagos",
#     "u. alberto hurtado":                               "universidad alberto hurtado",
# }

# # Normalización y mapeo
# ind_CNED["ies_norm"] = (
#     ind_CNED["Nombre Institución"]
#     .apply(normalizar_nombre_ies)
#     .map(lambda x: ind_CNED_fix.get(x, x))
# )
# ind_CNED["ies"] = ind_CNED["ies_norm"].map(abreviaciones_ies).fillna(ind_CNED["ies_norm"])

# ind_CNED["Año"] = pd.to_numeric(ind_CNED["Año"], errors="coerce").astype("Int64")


# # Filtro por año y por IES reconocidas en el mapa
# periodo = 2025
# ind_CNED_f = ind_CNED[
#     (ind_CNED["Año"] == periodo) &
#     (ind_CNED["ies_norm"].isin(ind_CNED_fix.values()))].copy()


# # =============================================================================
# # acá voy
# # =============================================================================


# # ── 1. Descomponer Códgo SIES en merged_final (5 keys) ──────────────────
# #   I = institución | S = sede | C = carrera | J = jornada | V = versión
# pattern = r'I(\d+)S(\d+)C(\d+)J(\d+)V(\d+)'

# keys_parsed = merged_final["código carrera"].str.extract(pattern)
# keys_parsed.columns = ["key_I", "key_S", "key_C", "key_J", "key_V"]
# keys_parsed = keys_parsed.astype("Int64")

# merged_final = pd.concat([merged_final, keys_parsed], axis=1)

# # ── 2. Aplicar la misma descomposición en ind_CNED ───────────────────────
# keys_parsed_cned = ind_CNED_f["Códgo SIES"].str.extract(pattern)
# keys_parsed_cned.columns = ["key_I", "key_S", "key_C", "key_J", "key_V"]
# keys_parsed_cned = keys_parsed_cned.astype("Int64")

# ind_CNED_f = pd.concat([ind_CNED_f.reset_index(drop=True),
#                          keys_parsed_cned.reset_index(drop=True)], axis=1)

# # ── 3. Merge: merged_final ← ind_CNED (todas las cols salvo las excluidas) ──
# keys_5 = ["key_I", "key_S", "key_C", "key_J", "key_V"]

# # Columnas a excluir de CNED (redundantes o ya presentes en merged_final)
# cols_excluir = {
#     "Nombre Institución",
#     "Tipo Institución",
#     "ies_norm",
#     "ies",
#     "Año",
#     "Códgo SIES",
# }

# cols_cned = [c for c in ind_CNED_f.columns if c not in cols_excluir]
# cned_slim = ind_CNED_f[cols_cned].drop_duplicates(subset=keys_5)

# merged_con_cned = merged_final.merge(
#     cned_slim,
#     on=keys_5,
#     how="left",
#     indicator=True,
# )

# # ── 4. Flag de presencia CNED ─────────────────────────────────────────────
# merged_con_cned["presencia_CNED"] = (merged_con_cned["_merge"] == "both").astype(int)
# merged_con_cned = merged_con_cned.drop(columns="_merge")

# # ── 5. Diagnóstico ────────────────────────────────────────────────────────
# n_total    = len(merged_con_cned)
# n_match    = merged_con_cned["presencia_CNED"].sum()
# n_no_match = n_total - n_match

# print(f"Total filas:        {n_total}")
# print(f"Con match CNED:     {n_match}  ({n_match/n_total:.1%})")
# print(f"Sin match CNED:     {n_no_match}  ({n_no_match/n_total:.1%})")

# print("\n── Match rate por Nivel Global ──")
# print(
#     merged_con_cned
#     .groupby("nivel global")["presencia_CNED"]
#     .agg(["sum", "count"])
#     .assign(match_rate=lambda x: x["sum"] / x["count"])
#     .rename(columns={"sum": "con_match", "count": "total"})
#     .sort_values("match_rate", ascending=False)
#     .to_string()
# )
# %% Retención - Min

df_ret = pd.read_csv('datos/Retención/trayectoria_universidades.csv')


cohort_2024 = df_ret[df_ret['año_ingreso'] == 2024].copy()

# Aggregate to codigo_unico level: retention rate from Ret_2024 → Ret_2025
ret_1er_ano = (
    cohort_2024
    .groupby('codigo_unico', as_index=False)
    .agg(
        nivel_global=('nivel_global', 'first'),
        n_2024=('Ret_2024', 'sum'),
        n_2025=('Ret_2025', 'sum'),
    )
)

ret_1er_ano['ret_1er_año'] = ret_1er_ano['n_2025'] / ret_1er_ano['n_2024']

ret_1er_ano['ret_1er_año_q'] = pd.qcut(ret_1er_ano['ret_1er_año'], q=4, labels=['Q1','Q2','Q3','Q4'])

tab(ret_1er_ano['nivel_global'])

ret_1er_ano[keys_5] = (
    ret_1er_ano['codigo_unico']
    .str.extract(r'I(\d+)S(\d+)C(\d+)J(\d+)V(\d+)')
    .astype('Int64')
)

merged_final = merged_final.merge(
    ret_1er_ano[keys_5 + ['ret_1er_año']],
    on=keys_5,
    how='left',
    indicator=True,
)

merged_final['flag_retencion'] = (merged_final['_merge'] == 'both').astype(int)
merged_final = merged_final.drop(columns='_merge')

print(f"Con retención:    {merged_final['flag_retencion'].sum()}")
print(f"Sin retención:    {(merged_final['flag_retencion'] == 0).sum()}")

tab(merged_final['flag_matricula'], merged_final['flag_retencion'])



# %% Buscador Empleabilidad - SIES
#
# Esta fuente NO trae código único (ni sede/jornada/versión) — sólo publica a
# nivel institución × programa. El match contra merged_final (grano keys_5) se
# hace entonces un nivel más agregado: (cod_inst, nombre_carrera_normalizado).
# Un valor de esta fuente se replica por lo tanto sobre todas las filas de
# merged_final que comparten institución+programa pero difieren en sede,
# jornada o versión (fan-out medido: media 2.4, máx 20 filas). Ese fan-out
# nunca cruza APDA (verificado: todo grupo con fan-out queda dentro de una sola
# APDA), así que el único efecto aguas abajo es de ponderación dentro de la
# celda IES×APDA — resuelto agregando por matrícula en
# Utils/data_prep.py::aggregate_ies_apda_rich en lugar de promedio simple.
# Ver "Inspection Empleabilidad/" para el análisis completo de cobertura.
#
# La hoja por defecto (Hoja1) es una tabla dinámica, no los datos — usar la
# hoja "Carreras e IES (2025-2026)".
#
# Además del match a merged_final, se engancha aquí el peso de titulados por
# carrera genérica (Data/catalogos/titulados_generica.csv, generado por
# Exp_Titulados_Generica.py) que usa Utils/data_prep.py::
# add_empleabilidad_ponderada para construir empleabilidad_pond_1er_año/2do_año
# — una alternativa a la ponderación por matrícula que pesa cada carrera
# genérica de la celda IES×APDA por su volumen real de egresados en la
# ventana de años relevante, en vez de por su matrícula total. Se engancha acá
# porque (Código, Nombre carrera genérica) es único en esta fuente — no hace
# falta pasar por el fan-out de programa.

bsc_emp = pd.read_excel(
    'datos/SIES/Buscador_Empleabilidad_ingresos_2025_2026_SIES (1).xlsx',
    sheet_name='Carreras e IES (2025-2026)',
    header=0,
)

# Filas de pie de página (fuente, totales) sin Código
bsc_emp = bsc_emp[bsc_emp['Código'].notna()].copy()
bsc_emp['cod_inst'] = bsc_emp['Código'].astype(int)

# Correcciones puntuales para programas cuyo nombre en esta fuente no coincide
# tras normalizar. La mayoría de los 182 programas sin match (de 1,190) no son
# corregibles así: SIES suele fusionar en una sola fila varios títulos que
# master_dataset mantiene como programas separados (menciones, diplomas,
# dobles titulaciones) — un dict 1:1 no puede resolver eso sin adivinar cuál
# programa es. Sólo se listan aquí los casos verificados de correspondencia
# 1:1 inequívoca (typo de la fuente o título abreviado/expandido).
_carrera_fix = {
    (20, 'ingeniero en marina mercante y transporte maritimo'): 'ingeniero en marina mercante',
    (86, 'pedagogia en educacion mediaen matematica'): 'pedagogia en educacion media en matematica',
}


def _prog_norm_empleabilidad(row):
    raw = normalizar_nombre_carrera(row['Nombre carrera (del título)'])
    return _carrera_fix.get((row['cod_inst'], raw), raw)


bsc_emp['_prog'] = bsc_emp.apply(_prog_norm_empleabilidad, axis=1)
bsc_emp = bsc_emp.drop_duplicates(subset=['cod_inst', '_prog'])

bsc_emp = bsc_emp.rename(columns={
    'Empleabilidad 1er año':                      'empleabilidad_1er_año',
    'Empleabilidad 2° año':                        'empleabilidad_2do_año',
    '% titulados con continuidad de estudios':     'continuidad_estudios',
})
_emp_cols = ['empleabilidad_1er_año', 'empleabilidad_2do_año', 'continuidad_estudios']
for c in _emp_cols:
    bsc_emp[c] = pd.to_numeric(bsc_emp[c], errors='coerce')

# Peso de titulados por carrera genérica (ver Exp_Titulados_Generica.py). Se
# engancha acá, antes del merge a merged_final, porque (cod_inst, Nombre
# carrera genérica) es único en bsc_emp — no pasa por el fan-out de programa.
_tit_gen = pd.read_csv('Data/catalogos/titulados_generica.csv', encoding='utf-8-sig')
_w_cols = ['titulados_gen_emp1', 'titulados_gen_emp2']
bsc_emp = bsc_emp.merge(
    _tit_gen,
    left_on=['cod_inst', 'Nombre carrera genérica'],
    right_on=['cod_inst', 'carrera_generica'],
    how='left',
)
_n_con_peso = bsc_emp['titulados_gen_emp1'].notna().sum()
print(f"\nEmpleabilidad SIES con peso de titulados (genérica): "
      f"{_n_con_peso} / {len(bsc_emp)} ({_n_con_peso / len(bsc_emp):.1%})")

merged_final['_prog'] = merged_final['nombre carrera'].apply(normalizar_nombre_carrera)

merged_final = merged_final.merge(
    bsc_emp[['cod_inst', '_prog'] + _emp_cols + _w_cols],
    on=['cod_inst', '_prog'],
    how='left',
    indicator=True,
)
merged_final['flag_empleabilidad'] = (merged_final['_merge'] == 'both').astype(int)
merged_final = merged_final.drop(columns=['_merge', '_prog'])

print(f"Con empleabilidad: {merged_final['flag_empleabilidad'].sum()}")
print(f"Sin empleabilidad: {(merged_final['flag_empleabilidad'] == 0).sum()}")
print("\n=== flag_empleabilidad × nivel global ===")
print(pd.crosstab(merged_final['nivel global'], merged_final['flag_empleabilidad']))

# Diagnóstico: filas de la fuente que no encontraron programa en merged_final
# (mismo patrón que Exp_SCIVAL.py — instituciones/programas sin match esperado)
_mf_pairs = set(zip(
    merged_final['cod_inst'],
    merged_final['nombre carrera'].apply(normalizar_nombre_carrera),
))
_sin_match_emp = bsc_emp[~bsc_emp.apply(
    lambda r: (r['cod_inst'], r['_prog']) in _mf_pairs, axis=1
)]
print(f"\n── Empleabilidad SIES: programas sin match en merged_final: "
      f"{len(_sin_match_emp)} / {len(bsc_emp)} ──")


# %% Buscador EstadísticasCarrera - SIES
#
# Descartada: esta fuente no trae institución (grano = Área × Tipo de
# institución × Carrera genérica, 252 filas). Es un benchmark nacional, no una
# fuente IES-diferenciadora — no aporta nada que Empleabilidad no traiga ya
# a nivel institución. Ver "Inspection Empleabilidad/" para el detalle.


# %% JCE weights

# PAC provides JCE/academics at the IES level only. Program-level proxies,
# all allocating an institution-constant count down to each program via a
# shared _allocator (see below) — the only allocator the data supports, no
# source breaks JCE/academics down by discipline/APDA. Revised 2026-07-28
# (see "Inspection JCE/" for the full before/after analysis) to fix two
# separate problems that both lived inside that one allocator shape:
#   (a) Postítulo matrícula was inflating every institution's matrícula base
#       despite Postítulo not being a degree-granting level — now excluded
#       from both the numerator and denominator of _allocator. Postítulo
#       program rows get NaN on every weight below, but stay in the dataset
#       for every other (non-JCE) dimension.
#   (b) the own-institution matrícula share systematically inflated w_mat/
#       w_jce_*/w_academicos_39 for institutions offering fewer APDAs (all
#       their JCE gets allocated across a smaller area footprint) — now
#       damped by dividing by _n_apda_activas (count of APDAs where the
#       institution has non-Postítulo matrícula > 0). Partial, not complete,
#       fix — see the dated doc for the measured effect and the residual
#       size/breadth confound that no allocator can fully remove.
#
# w_mat  — total IES JCE × _allocator
#           Baseline; academic effort follows realised student demand.
#
# w_doc  — Doctor JCE × (program postgrad matrícula / IES postgrad matrícula)
#           Allocates only PhD-level staff to postgrad programs by their
#           share of the university's postgrad enrollment. NaN for pregrado.
#           Unchanged by the 2026-07-28 revision — Posgrado already excludes
#           Postítulo by construction, and this denominator is scoped
#           narrower than the shared _allocator to begin with.
#
# w_post — (Doctor + Magister + Especialidad médica) JCE × _allocator
#           Like w_mat but restricted to postgrad-related staff JCE. Not
#           wired into either config.yaml's dimensions (retired in favour of
#           w_jce_doctor/magister/especialidad below); kept internally
#           consistent with _allocator anyway.
#
# w_jce_doctor / w_jce_magister / w_jce_especialidad — composition ratio
#           (JCE del tipo / JCE total institucional) × _allocator. Until
#           2026-07-28 this was the raw JCE-by-formación count (not a ratio)
#           × _allocator, which conflated "how many doctors" with "how big
#           is this institution" into one undifferentiated mass — see the
#           dated doc (report Section 5) for the before/after comparison.
#           Because the numerator is now a ratio, w_jce_doctor + w_jce_
#           magister + w_jce_especialidad no longer equals w_post
#           (pre-normalisation) — expected, not a bug.
#           Retired from config.yaml scoring 2026-08-12: even the ratio
#           numerator (composición real) survives, _allocator still fabricates
#           within-IES variance that no source supports — see "Inspection
#           JCE/" Sections 3-9. Left computed here (and in merged_final.csv)
#           only because Exp_JCE_casos.py's diagnostic report depends on this
#           exact column as its documented baseline.
#
# pct_jce_doctor / pct_jce_magister / pct_jce_especialidad — the same ratio
#           (JCE del tipo / JCE total institucional), without _allocator: a
#           single value per IES, repeated across every APDA row of that
#           institution. Replaces w_jce_doctor/magister/especialidad in
#           config.yaml scoring 2026-08-12 — honest about what the source
#           data actually supports (institution-level composition only, no
#           real APDA breakdown exists anywhere upstream of PAC).
#
# w_academicos_39 — N° académicos con 39+ horas contratadas (BD_Académicos_
#           Número) × _allocator. Same formula as w_mat, applied to headcount
#           instead of JCE. Kept as an independent variable alongside
#           w_jce_doctor despite their 0.90 correlation — JCE is a reduced
#           form of academic headcount (FTE vs. raw person-count), so
#           correlation is structurally expected and both carry information
#           (2026-07-28 decision, no code change from this reasoning alone).
#           Retired from config.yaml scoring 2026-08-13, same _allocator
#           rationale as w_jce_doctor/magister/especialidad above. Left
#           computed here (and in merged_final.csv) for continuity.
#
# pct_academicos_39 — % del cuerpo académico institucional con 39+ horas
#           contratadas, without _allocator: a single value per IES, repeated
#           across every APDA row of that institution. Replaces
#           w_academicos_39 in config.yaml scoring 2026-08-13, same rationale
#           as pct_jce_doctor/magister/especialidad — institution-level
#           composition only, no real APDA breakdown exists upstream of PAC.
#
# pct_jce_1inst — % of the institution's JCE held by academics exclusive to
#           that institution (don't split their JCE with another university),
#           without _allocator: a single value per IES, repeated across every
#           APDA row. Replaces the raw headcount column
#           n_de_jce_de_la_institucion_que_trabajan_en_1_o_mas_instituciones_
#           en_1_institucion in config.yaml scoring 2026-08-13 — the raw count
#           scales with institution size, not exclusivity, so bigger
#           institutions won on headcount even when a smaller share of their
#           JCE was actually exclusive. Same "no _allocator, ratio only"
#           rationale as pct_jce_doctor/magister/especialidad/academicos_39.
#
# jce_por_estudiantes — total institutional JCE / total institutional
#           matrícula (excl. Postítulo, but keeping Especialidad Médica u
#           Odontológica, which is a `nivel carrera` value that
#           `nivel global == 'Postítulo'` would otherwise wrongly absorb —
#           see Inspection JCE/Exp_JCE_casos.py). Raw ratio (not ×100),
#           institution-constant. Staffing-intensity counterpart to the
#           pct_jce_*/pct_academicos_39 composition ratios above — added
#           2026-08-13.

doc_col = 'n_de_jce_por_nivel_de_formacion_doctor'
mag_col = 'n_de_jce_por_nivel_de_formacion_magister'
esp_col = 'n_de_jce_por_nivel_de_formacion_especialidad_medica_u_odontologica'
excl_col = 'n_de_jce_de_la_institucion_que_trabajan_en_1_o_mas_instituciones_en_1_institucion'

_mat_post_prog = merged_final['total matrícula'].where(
    merged_final['nivel global'] == 'Posgrado'
)
_ies_mat_post = merged_final.groupby('ies_norm')['total matrícula'].transform(
    lambda x: x[merged_final.loc[x.index, 'nivel global'] == 'Posgrado'].sum()
)

# _allocator — shared by w_mat/w_post/w_jce_*/w_academicos_39 (not w_doc,
# which keeps its own posgrado-restricted denominator above): institution's
# matrícula share of the program's APDA, excluding Postítulo from both sides
# of the share, damped by the institution's own APDA breadth.
_mat_no_post = merged_final['total matrícula'].where(merged_final['nivel global'] != 'Postítulo')
_ies_mat = merged_final.groupby('ies_norm')['total matrícula'].transform(
    lambda x: x[merged_final.loc[x.index, 'nivel global'] != 'Postítulo'].sum(min_count=1)
)

_apda_mat_sum_no_post = (
    merged_final[(merged_final['apda'] != 'No aplica') & (merged_final['nivel global'] != 'Postítulo')]
    .groupby(['ies_norm', 'apda'])['total matrícula']
    .sum(min_count=1)
)
_n_apda_activas = _apda_mat_sum_no_post[_apda_mat_sum_no_post > 0].groupby('ies_norm').size()
merged_final['_n_apda_activas'] = merged_final['ies_norm'].map(_n_apda_activas)

_allocator = (_mat_no_post / _ies_mat) / merged_final['_n_apda_activas']

merged_final['w_mat'] = merged_final['n_de_jce_por_institucion_total_general'] * _allocator

merged_final['w_doc'] = (
    merged_final[doc_col]
    * (_mat_post_prog / _ies_mat_post)
)

_post_jce = merged_final[[doc_col, mag_col, esp_col]].sum(axis=1, min_count=1)
merged_final['w_post'] = _post_jce * _allocator

_jce_total_col = merged_final['n_de_jce_por_institucion_total_general']
merged_final['w_jce_doctor'] = (merged_final[doc_col] / _jce_total_col) * _allocator
merged_final['w_jce_magister'] = (merged_final[mag_col] / _jce_total_col) * _allocator
merged_final['w_jce_especialidad'] = (merged_final[esp_col] / _jce_total_col) * _allocator

merged_final['pct_jce_doctor'] = merged_final[doc_col] / _jce_total_col * 100
merged_final['pct_jce_magister'] = merged_final[mag_col] / _jce_total_col * 100
merged_final['pct_jce_especialidad'] = merged_final[esp_col] / _jce_total_col * 100

merged_final['w_academicos_39'] = merged_final[_col_39] * _allocator

merged_final['pct_academicos_39'] = merged_final[_col_39] / merged_final[_col_academicos_total] * 100

merged_final['pct_jce_1inst'] = merged_final[excl_col] / _jce_total_col * 100

_is_esp_medica = merged_final['nivel carrera'].str.contains('Especialidad', case=False, na=False)
_mat_keep_esp_mask = (merged_final['nivel global'] != 'Postítulo') | _is_esp_medica
_ies_mat_jce = merged_final.groupby('ies_norm')['total matrícula'].transform(
    lambda x: x[_mat_keep_esp_mask.loc[x.index]].sum(min_count=1)
)
merged_final['jce_por_estudiantes'] = _jce_total_col / _ies_mat_jce

# Normalise to 0-100: top-ranked IES = 100
_all_weights = ['w_mat', 'w_doc', 'w_post', 'w_jce_doctor', 'w_jce_magister',
                 'w_jce_especialidad', 'w_academicos_39']
for _w in _all_weights:
    merged_final[_w] = merged_final[_w] / merged_final[_w].max() * 100

for _w in _all_weights:
    print(f"{_w:<20} — non-null: {merged_final[_w].notna().sum()}  |  NaN: {merged_final[_w].isna().sum()}")

for _w in ['pct_jce_doctor', 'pct_jce_magister', 'pct_jce_especialidad', 'pct_academicos_39', 'pct_jce_1inst', 'jce_por_estudiantes']:
    print(f"{_w:<20} — non-null: {merged_final[_w].notna().sum()}  |  NaN: {merged_final[_w].isna().sum()}")


# %%% JCE weights — completeness

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

_weights   = ['w_mat', 'w_doc', 'w_post']
_niveles   = ['Pregrado', 'Posgrado']
_n_total   = len(merged_final)

# --- data for plots ---

# 1. Overall completeness (% non-null) per weight
overall = {w: merged_final[w].notna().mean() * 100 for w in _weights}

# 2. Completeness by nivel global × weight
by_nivel = (
    merged_final
    .groupby('nivel global')[_weights]
    .apply(lambda g: g.notna().mean() * 100)
    .loc[lambda d: d.index.isin(_niveles)]   # keep only Pre/Pos
)

# 3. % of programs with ALL three weights non-null, by APDA
complete_row = merged_final[_weights].notna().all(axis=1)
by_apda = (
    merged_final
    .assign(_complete=complete_row)
    .groupby('apda')['_complete']
    .mean() * 100
).sort_values()

# --- figure ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1 — overall bars
ax = axes[0]
bars = ax.bar(_weights, [overall[w] for w in _weights],
              color=['#2196F3', '#4CAF50', '#FF9800'], width=0.5)
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title('Overall completeness\n(% non-null)')
ax.set_xlabel('Weight')
for bar, w in zip(bars, _weights):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{overall[w]:.1f}%', ha='center', va='bottom', fontsize=9)

# Panel 2 — completeness by nivel global
ax2 = axes[1]
x   = range(len(_weights))
w2  = 0.35
colors_niv = {'Pregrado': '#90CAF9', 'Posgrado': '#1565C0'}
for i, niv in enumerate(_niveles):
    vals = [by_nivel.loc[niv, w] if niv in by_nivel.index else 0 for w in _weights]
    ax2.bar([xi + i * w2 for xi in x], vals, width=w2,
            label=niv, color=colors_niv[niv])
ax2.set_xticks([xi + w2 / 2 for xi in x])
ax2.set_xticklabels(_weights)
ax2.set_ylim(0, 110)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_title('Completeness by nivel global\n(% non-null per weight)')
ax2.legend()

# Panel 3 — % fully complete (all weights) by APDA
ax3 = axes[2]
ax3.barh(by_apda.index, by_apda.values, color='#546E7A')
ax3.xaxis.set_major_formatter(mtick.PercentFormatter())
ax3.set_title('Programs with all 3 weights\nnon-null, by APDA (%)')
ax3.set_xlabel('% programs fully covered')
for i, (val, label) in enumerate(zip(by_apda.values, by_apda.index)):
    ax3.text(val + 0.5, i, f'{val:.0f}%', va='center', fontsize=8)

plt.suptitle('JCE weight completeness', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# %%% JCE weights — completeness (flag_matricula == 1)

_df = merged_final[merged_final['flag_matricula'] == 1]

overall = {w: _df[w].notna().mean() * 100 for w in _weights}

by_nivel = (
    _df
    .groupby('nivel global')[_weights]
    .apply(lambda g: g.notna().mean() * 100)
    .loc[lambda d: d.index.isin(_niveles)]
)

complete_row = _df[_weights].notna().all(axis=1)
by_apda = (
    _df
    .assign(_complete=complete_row)
    .groupby('apda')['_complete']
    .mean() * 100
).sort_values()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
bars = ax.bar(_weights, [overall[w] for w in _weights],
              color=['#2196F3', '#4CAF50', '#FF9800'], width=0.5)
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title('Overall completeness\n(% non-null)')
ax.set_xlabel('Weight')
for bar, w in zip(bars, _weights):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{overall[w]:.1f}%', ha='center', va='bottom', fontsize=9)

ax2 = axes[1]
x  = range(len(_weights))
w2 = 0.35
for i, niv in enumerate(_niveles):
    vals = [by_nivel.loc[niv, w] if niv in by_nivel.index else 0 for w in _weights]
    ax2.bar([xi + i * w2 for xi in x], vals, width=w2,
            label=niv, color=colors_niv[niv])
ax2.set_xticks([xi + w2 / 2 for xi in x])
ax2.set_xticklabels(_weights)
ax2.set_ylim(0, 110)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_title('Completeness by nivel global\n(% non-null per weight)')
ax2.legend()

ax3 = axes[2]
ax3.barh(by_apda.index, by_apda.values, color='#546E7A')
ax3.xaxis.set_major_formatter(mtick.PercentFormatter())
ax3.set_title('Programs with all 3 weights\nnon-null, by APDA (%)')
ax3.set_xlabel('% programs fully covered')
for i, (val, label) in enumerate(zip(by_apda.values, by_apda.index)):
    ax3.text(val + 0.5, i, f'{val:.0f}%', va='center', fontsize=8)

plt.suptitle(f'JCE weight completeness — flag_matricula == 1 (n={len(_df):,})',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# %% Drop variables

cols_drop = [
    'elegibilidad beca pedagogía',
    'pedagogía medicina odontología, otro',
    'área dest agricultura',
    'área dest c soc comerc derecho',
    'área dest ciencias',
    'área dest educación',
    'área dest humanidades y artes',
    'área dest ing ind const',
    'área dest salud y serv soc',
    'área dest servicios',
]

merged_final = merged_final.drop(columns=[c for c in cols_drop if c in merged_final.columns])

# %% Export

ins(merged_final, "flag_matricula")
tab(merged_final['flag_matricula']) 

merged_final.to_csv('Data/merged_final.csv', index=False, encoding='utf-8-sig')
# merged_con_cned.to_csv('Data/merged_con_cned.csv', index=False, encoding='utf-8-sig')
in_of_not_mat.to_csv('Data/in_of_not_mat.csv', index=False, encoding='utf-8-sig')

print(f"merged_final guardado:   {len(merged_final)} filas")
# print(f"merged_con_cned guardado:{len(merged_con_cned)} filas")
print(f"in_of_not_mat guardado:  {len(in_of_not_mat)} filas")
