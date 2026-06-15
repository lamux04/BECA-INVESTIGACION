"""Preparacion de datasets tabulares para modelos T5."""

import pandas as pd


SOURCE_TEXT_COL = "source_text"
TARGET_TEXT_COL = "target_text"


def dataframe_a_source_target(df, label_col="LABEL", separator="|"):
    """Convierte features tabulares y etiqueta al formato textual usado por T5."""
    if label_col not in df.columns:
        raise KeyError(f"La columna de etiquetas no existe: {label_col}")

    if not separator:
        raise ValueError("separator no puede estar vacio.")

    feature_cols = [column for column in df.columns if column != label_col]

    if not feature_cols:
        raise ValueError("El dataset debe contener al menos una feature.")

    features = df[feature_cols].astype(str)

    return pd.DataFrame(
        {
            SOURCE_TEXT_COL: features.agg(separator.join, axis=1),
            TARGET_TEXT_COL: df[label_col].astype(str),
        }
    )


def validar_source_target(df, dataset_name="dataset"):
    """Valida que un DataFrame tenga el formato textual requerido por T5."""
    required_cols = {SOURCE_TEXT_COL, TARGET_TEXT_COL}
    missing_cols = required_cols.difference(df.columns)

    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise KeyError(f"Faltan columnas en {dataset_name}: {missing}")

    if df.empty:
        raise ValueError(f"{dataset_name} no puede estar vacio.")

