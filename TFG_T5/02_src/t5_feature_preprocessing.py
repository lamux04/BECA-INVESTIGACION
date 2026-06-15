"""Preprocesamiento de features tabulares para los experimentos T5."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler


def _transformar_scaler_pca(
    datasets,
    label_col,
    n_components,
    scaler,
    scaler_name,
    random_state=42,
):
    """Ajusta scaler y PCA con train y transforma train, validacion y test."""
    train_features = [
        column for column in datasets["train"].columns if column != label_col
    ]

    if not train_features:
        raise ValueError("El dataset de entrenamiento no contiene features.")

    for name, df in datasets.items():
        if label_col not in df.columns:
            raise KeyError(f"Falta la columna de etiquetas en {name}: {label_col}")

        feature_cols = [column for column in df.columns if column != label_col]
        if feature_cols != train_features:
            raise ValueError(
                f"Las features de {name} no coinciden con las de train."
            )

        non_numeric = df[train_features].select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            raise TypeError(
                f"Todas las features deben ser numericas en {name}: "
                + ", ".join(non_numeric)
            )

    if n_components > min(len(datasets["train"]), len(train_features)):
        raise ValueError(
            f"PCA con k={n_components} no es posible para "
            f"{len(datasets['train'])} filas y {len(train_features)} features."
        )

    train_scaled = scaler.fit_transform(datasets["train"][train_features])

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(train_scaled)

    scaled_columns = [f"scaled__{column}" for column in train_features]
    pca_columns = [f"pca__{index:02d}" for index in range(1, n_components + 1)]
    transformed = {}

    for name, df in datasets.items():
        scaled_values = scaler.transform(df[train_features])
        pca_values = pca.transform(scaled_values)
        feature_values = np.hstack([scaled_values, pca_values])
        transformed_df = pd.DataFrame(
            feature_values,
            columns=scaled_columns + pca_columns,
            index=df.index,
        )
        transformed_df[label_col] = df[label_col].to_numpy()
        transformed[name] = transformed_df.reset_index(drop=True)

    metadata = {
        "preprocessing": f"{scaler_name} + PCA + concatenation",
        "original_features": len(train_features),
        "pca_components": n_components,
        "transformed_features": len(scaled_columns) + len(pca_columns),
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
    }
    return transformed, scaler, pca, metadata


def transformar_standard_scaler_pca(
    datasets,
    label_col,
    n_components,
    random_state=42,
):
    """Concatena features estandarizadas y PCA, ajustados solo con train."""
    return _transformar_scaler_pca(
        datasets=datasets,
        label_col=label_col,
        n_components=n_components,
        scaler=StandardScaler(),
        scaler_name="StandardScaler",
        random_state=random_state,
    )


def transformar_robust_scaler_pca(
    datasets,
    label_col,
    n_components,
    random_state=42,
):
    """Concatena features con RobustScaler y PCA, ajustados solo con train."""
    return _transformar_scaler_pca(
        datasets=datasets,
        label_col=label_col,
        n_components=n_components,
        scaler=RobustScaler(),
        scaler_name="RobustScaler",
        random_state=random_state,
    )
