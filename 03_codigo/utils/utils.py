import pandas as pd
import glob
import os
import numpy as np


def cargar_dataset(nombre_dataset, ruta_base="../../02_datasets/raw"):
    """
    Lee todos los CSV de un dataset y los concatena en un solo DataFrame.

    Parameters
    ----------
    nombre_dataset : str
        Nombre del dataset (ej: 'CIC17')
    ruta_base : str
        Ruta base donde están los datasets

    Returns
    -------
    df : pandas.DataFrame
        DataFrame concatenado con todos los CSV
    """

    directorio_actual = os.getcwd()

    directorio_dataset = os.path.abspath(
        os.path.join(directorio_actual, ruta_base, nombre_dataset)
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

import os

def guardar_dataset_csv(df, nombre_archivo, ruta="../../02_datasets/clean"):

    # crear carpeta si no existe
    os.makedirs(ruta, exist_ok=True)

    ruta_completa = os.path.join(ruta, nombre_archivo)

    df.to_csv(ruta_completa, index=False)