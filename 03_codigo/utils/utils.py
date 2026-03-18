import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE


def cargar_dataset(nombre_dataset = "*", ruta_base="../../02_datasets/raw"):
    """
    Si nombre_dataset == *, concatena todos los datasets de la carpeta.
    Sino lee solo el .csv indicado

    Parameters
    ----------
    nombre_dataset : str
        Nombre del dataset (ej: 'CIC17.csv')
    ruta_base : str
        Ruta base donde están los datasets

    Returns
    -------
    df : pandas.DataFrame
        DataFrame concatenado con todos los CSV
    """

    # Leer todos los datasets de la carpeta y concatenarlos
    if nombre_dataset == "*":
        directorio_actual = os.getcwd()

        directorio_dataset = os.path.abspath(
            os.path.join(directorio_actual, ruta_base)
        )

        archivos = glob.glob(os.path.join(directorio_dataset, "*.csv"))
        archivos.sort()

        df_list = []

        for archivo in archivos:
            df_temp = pd.read_csv(
                archivo,
                low_memory=False,
                on_bad_lines="warn"
            )
            df_list.append(df_temp)
    
        df = pd.concat(df_list, ignore_index=True)

    # Leer un dataset
    else:
        directorio_actual = os.getcwd()
        archivo_dataset = os.path.abspath(
            os.path.join(directorio_actual, ruta_base, nombre_dataset)
        )
        df = pd.read_csv(
            archivo_dataset, 
            low_memory=False, 
            on_bad_lines="warn"
        )

    return df

def homogeneizar_columnas(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.upper()
    )
    
    return df

def eliminar_columnas_constantes(df):
    return df.loc[:, df.nunique() > 1]

def eliminar_columnas(df, to_drop):
    return df.drop(columns=to_drop)

def eliminar_columnas_altamente_correlacionadas(df, threshold, label_col="LABEL"):

    # separar temporalmente
    X = df.drop(columns=label_col)
    y = df[label_col]

    corr_matrix = X.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    X = X.drop(columns=to_drop)

    # volver a juntar
    df_clean = pd.concat([X, y], axis=1)

    return df_clean


def eliminar_infinitos_vacios_nulos(df):
    
    # reemplazar infinitos por NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # eliminar filas con NaN
    df = df.dropna()
    
    # eliminar strings vacíos si los hubiera
    df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
    
    return df

def eliminar_filas_duplicadas(df):
    df = df.drop_duplicates()
    return df

def guardar_dataset_csv(df, nombre_archivo, ruta):

    # crear carpeta si no existe
    os.makedirs(ruta, exist_ok=True)

    ruta_completa = os.path.join(ruta, nombre_archivo)

    df.to_csv(ruta_completa, index=False)


def codificar_etiqueta_label(df, etiqueta):
    # obtener distribución de clases
    counts = df[etiqueta].value_counts()

    # crear lista de etiquetas ordenadas
    labels = list(counts.index)

    # asegurar que Benign sea 0
    if "Benign" in labels:
        labels.remove("Benign")
        labels = ["Benign"] + labels

    # crear mapping
    mapping = {label: i for i, label in enumerate(labels)}

    # aplicar codificación
    df[etiqueta] = df[etiqueta].map(mapping)

    return df, mapping

def codificar_etiqueta_label(df, etiqueta):
    
    # obtener clases ordenadas por frecuencia
    labels = df[etiqueta].value_counts().index.tolist()

    # crear mapping automático
    mapping = {label: i for i, label in enumerate(labels)}

    # aplicar codificación
    df[etiqueta] = df[etiqueta].map(mapping)

    return df, mapping

def codificar_columnas_string(df, label_col="LABEL"):
    
    for col in df.columns:
        if col != label_col and (
            pd.api.types.is_object_dtype(df[col]) or
            pd.api.types.is_string_dtype(df[col])
        ):
            df[col], _ = pd.factorize(df[col])

    return df

def dividir_train_test_stratified(df, label_col="LABEL", test_size=0.2, random_state=42):
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )
    
    # resetear índices (recomendado)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, test_df

def dataframe_a_source_target(df, label_col="LABEL", sep="|"):
    
    # Separar X e y
    X = df.drop(columns=[label_col]).astype(str)
    y = df[label_col].astype(str)
    
    # Crear source_text (todas las features unidas)
    source_text = X.agg(sep.join, axis=1)
    
    # Crear DataFrame final
    df_final = pd.DataFrame({
        "source_text": source_text,
        "target_text": y
    })
    
    return df_final

def dividir_en_k_particiones_stratified(df, label_col="LABEL", k=5, random_state=42):
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    particiones = []
    
    for train_index, val_index in skf.split(X, y):
        
        df_train_fold = df.iloc[train_index].reset_index(drop=True)
        df_val_fold = df.iloc[val_index].reset_index(drop=True)
        
        particiones.append((df_train_fold, df_val_fold))
    
    return particiones


def undersample_clase_mayoritaria_rus(df, n_mayoritaria, label_col="LABEL", random_state=42):
    
    X = df.drop(columns=label_col)
    y = df[label_col]
    
    # encontrar la clase mayoritaria
    clase_mayoritaria = y.value_counts().idxmax()
    
    # diccionario que indica cuántas muestras queremos de esa clase
    sampling_strategy = {clase_mayoritaria: n_mayoritaria}
    
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    
    X_res, y_res = rus.fit_resample(X, y)
    
    df_res = pd.concat([X_res, y_res], axis=1)
    
    return df_res

def aplicar_enn(df, label_col="LABEL", n_neighbors=3, kind_sel="all"):
    """
    Aplica Edited Nearest Neighbours (ENN) a un DataFrame.

    Parámetros:
    - df: pandas DataFrame
    - label_col: nombre de la columna etiqueta
    - n_neighbors: número de vecinos para ENN
    - kind_sel: "all" o "mode"
        - "all": elimina una muestra si no coincide con todos sus vecinos
        - "mode": elimina una muestra si no coincide con la mayoría

    Devuelve:
    - df_res: DataFrame resultante tras aplicar ENN
    """

    # Separar variables y etiqueta
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Aplicar ENN
    enn = EditedNearestNeighbours(
        n_neighbors=n_neighbors,
        kind_sel=kind_sel
    )
    X_res, y_res = enn.fit_resample(X, y)

    # Reconstruir DataFrame
    df_res = pd.DataFrame(X_res, columns=X.columns)
    df_res[label_col] = y_res

    return df_res



def balancear_y_mover_a_test(
    df_train,
    df_test,
    label_col="LABEL",
    target_n=10000,
    random_state=42,
    k_neighbors=5
):
    
    df_train = df_train.copy()
    df_test = df_test.copy()

    clases = df_train[label_col].unique()

    # guardamos índices a eliminar
    indices_a_test = []

    for clase in clases:
        subset = df_train[df_train[label_col] == clase]

        if len(subset) > target_n:
            # muestreo aleatorio de los que se quedan
            subset_keep = subset.sample(n=target_n, random_state=random_state)

            # los que sobran se moverán a test
            subset_remove = subset.drop(subset_keep.index)

            indices_a_test.extend(subset_remove.index)

    # mover a test
    df_test = pd.concat([df_test, df_train.loc[indices_a_test]])

    # eliminar del train
    df_train = df_train.drop(indices_a_test)

    # ---------- SMOTE ----------
    
    X = df_train.drop(columns=[label_col])
    y = df_train[label_col]

    class_counts = y.value_counts()

    over_strategy = {
        cls: target_n
        for cls, count in class_counts.items()
        if count < target_n
    }

    if over_strategy:

        min_class_size = min(class_counts[cls] for cls in over_strategy)
        k = min(k_neighbors, min_class_size - 1)

        smote = SMOTE(
            sampling_strategy=over_strategy,
            random_state=random_state,
            k_neighbors=k
        )

        X_res, y_res = smote.fit_resample(X, y)

        df_train = pd.DataFrame(X_res, columns=X.columns)
        df_train[label_col] = y_res

    return df_train, df_test