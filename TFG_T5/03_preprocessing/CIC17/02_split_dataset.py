"""
Fichero de division del dataset limpio en train, test y folds train/validacion.

Ademas de guardar las particiones, este script genera reportes CSV con:

- Resumen de filas/columnas por particion.
- Distribucion de clases por particion.
- Parametros usados en la division.
"""

# ============== CONFIGURACIONES ================
DATASET_ORIGINAL = "CIC17"

# ============== LIBRERIAS =============
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "02_src"

sys.path.append(str(SRC_DIR))

import preprocessing
import reporting
import split

# ============== RUTAS =============
NOMBRE_DATASET_ENTRADA = f"{DATASET_ORIGINAL}__clean"
NOMBRE_DATASET_RESULTADO = f"{DATASET_ORIGINAL}__split"

RUTA_DATASET_ENTRADA = PROJECT_ROOT / "01_datasets" / DATASET_ORIGINAL / "clean"
RUTA_DATASET_SALIDA = PROJECT_ROOT / "01_datasets" / DATASET_ORIGINAL / "split"
RUTA_SALIDA_REPORTES = PROJECT_ROOT / "05_results" / DATASET_ORIGINAL / "split"

NOMBRE_DATASET_ENTRADA = f"{NOMBRE_DATASET_ENTRADA}.csv"
NOMBRE_TRAIN = f"{NOMBRE_DATASET_RESULTADO}__train.csv"
NOMBRE_TEST = f"{NOMBRE_DATASET_RESULTADO}__test.csv"

RUTA_DATASET_SALIDA.mkdir(parents=True, exist_ok=True)
RUTA_SALIDA_REPORTES.mkdir(parents=True, exist_ok=True)

# ============== PARAMETROS =============
LABEL_COL = "LABEL"
TEST_SIZE = 0.2
N_SPLITS = 5
RANDOM_STATE = 42

# ============== ESTRUCTURAS PARA REPORTES =============
split_summary = []
split_label_distribution = []
split_parameters = []


# ======================================================
# DIVISION DEL DATASET
# ======================================================

print("Cargando dataset limpio...")
df = preprocessing.cargar_dataset(
    nombre_dataset=NOMBRE_DATASET_ENTRADA,
    ruta_base=RUTA_DATASET_ENTRADA,
)

print("Dividiendo dataset en train y test...")
train_df, test_df = split.dividir_train_test_stratified(
    df=df,
    label_col=LABEL_COL,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

reporting.registrar_parametros_split(
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    label_col=LABEL_COL,
    n_splits=N_SPLITS,
)

reporting.registrar_resumen_split(
    df_original=df,
    train_df=train_df,
    test_df=test_df,
)

reporting.registrar_distribucion_clases_split(
    df_original=df,
    train_df=train_df,
    test_df=test_df,
    label_col=LABEL_COL,
)

# ============== GUARDAR DATASETS ================
print("Guardando datasets train y test...")

preprocessing.guardar_dataset_csv(
    df=train_df,
    nombre_archivo=NOMBRE_TRAIN,
    ruta=RUTA_DATASET_SALIDA,
)

preprocessing.guardar_dataset_csv(
    df=test_df,
    nombre_archivo=NOMBRE_TEST,
    ruta=RUTA_DATASET_SALIDA,
)

print(f"Generando y guardando {N_SPLITS} folds train/validacion...")

folds = split.iterar_folds_stratified(
    df=train_df,
    label_col=LABEL_COL,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
)

for fold_data in folds:
    fold = fold_data["fold"]

    for partition in ("train", "val"):
        nombre_particion = f"fold_{fold}_{partition}"
        df_particion = fold_data[partition]
        nombre_archivo = f"{NOMBRE_DATASET_RESULTADO}__{nombre_particion}.csv"

        preprocessing.guardar_dataset_csv(
            df=df_particion,
            nombre_archivo=nombre_archivo,
            ruta=RUTA_DATASET_SALIDA,
        )
        reporting.registrar_particion_split(
            df_partition=df_particion,
            partition=nombre_particion,
            reference_rows=len(df),
            label_col=LABEL_COL,
        )

# ============== GUARDAR REPORTES ================
print("Guardando reportes de division...")

reporting.guardar_reportes_split()

print("Division del dataset finalizada correctamente.")
print(f"Train guardado en: {RUTA_DATASET_SALIDA / NOMBRE_TRAIN}")
print(f"Test guardado en: {RUTA_DATASET_SALIDA / NOMBRE_TEST}")
print(f"Folds guardados en: {RUTA_DATASET_SALIDA}")
print(f"Reportes guardados en: {RUTA_SALIDA_REPORTES}")
