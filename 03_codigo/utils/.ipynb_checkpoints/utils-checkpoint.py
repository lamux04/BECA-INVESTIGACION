import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split


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