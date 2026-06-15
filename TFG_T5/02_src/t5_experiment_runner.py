"""Runner de fine-tuning y evaluacion de T5 para un fold concreto."""

from pathlib import Path
from dataclasses import asdict
import json
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import metrics
import preprocessing
import t5_data
import t5_feature_preprocessing
from t5_model import T5Classifier


def _guardar_distribucion_etiquetas(df, label_col, output_path):
    distribution = (
        df[label_col]
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("label")
        .reset_index(name="count")
    )
    distribution["proportion"] = distribution["count"] / len(df)
    distribution.to_csv(output_path, index=False)


def _guardar_matriz_confusion(y_true, y_pred, labels, csv_path, figure_path):
    unexpected = sorted(set(y_pred).difference(labels))
    display_labels = list(labels) + unexpected
    matrix = confusion_matrix(y_true, y_pred, labels=display_labels)

    df_matrix = pd.DataFrame(
        matrix,
        index=display_labels,
        columns=display_labels,
    )
    df_matrix.index.name = "true_label"
    df_matrix.columns.name = "predicted_label"
    df_matrix.to_csv(csv_path)

    figure_size = max(10, len(display_labels) * 0.7)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size * 0.8))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=display_labels,
    ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title("Matriz de confusion - test")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(figure_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _guardar_historial_entrenamiento(trainer, csv_path, figure_path):
    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False

    if "loss" in history.columns:
        train_rows = history.dropna(subset=["loss"])
        if not train_rows.empty:
            ax.plot(train_rows["epoch"], train_rows["loss"], marker="o", label="Train loss")
            plotted = True

    if "eval_loss" in history.columns:
        val_rows = history.dropna(subset=["eval_loss"])
        if not val_rows.empty:
            ax.plot(val_rows["epoch"], val_rows["eval_loss"], marker="s", label="Validation loss")
            plotted = True

    if plotted:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_path, format="pdf", bbox_inches="tight")

    plt.close(fig)


def ejecutar_experimento_t5(config):
    """Entrena un fold, selecciona el mejor epoch y evalua sobre test."""
    project_root = Path(config["project_root"])
    dataset = config["dataset"]
    fold = int(config["fold"])
    label_col = config.get("label_col", "LABEL")
    phase_dir_name = config.get("phase_dir_name", "fase_B_t5")
    phase_label = config.get("phase_label", "B_t5")
    experiment_name = config.get(
        "experiment_name_template",
        "B__t5_fold_{fold}",
    ).format(fold=fold)

    if fold not in range(1, 6):
        raise ValueError("fold debe estar entre 1 y 5.")

    split_dir = project_root / "01_datasets" / dataset / "split"
    model_base_path = Path(config["model_base_path"])
    result_dir = (
        project_root
        / "05_results"
        / dataset
        / phase_dir_name
        / experiment_name
    )
    report_dir = result_dir / "reports"
    figure_dir = result_dir / "figures"
    model_dir = (
        project_root
        / "07_models"
        / dataset
        / phase_dir_name
        / experiment_name
    )
    checkpoint_dir = model_dir / "checkpoints"
    best_model_dir = model_dir / "best_model"

    for path in (report_dir, figure_dir, checkpoint_dir, best_model_dir):
        path.mkdir(parents=True, exist_ok=True)

    input_files = {
        "train": f"{dataset}__split__fold_{fold}_train.csv",
        "validation": f"{dataset}__split__fold_{fold}_val.csv",
        "test": f"{dataset}__split__test.csv",
    }

    if not model_base_path.is_dir():
        raise FileNotFoundError(
            f"No se encontro el modelo T5 base en: {model_base_path}"
        )

    print(f"Cargando fold {fold} y test...")
    datasets = {
        name: preprocessing.cargar_dataset(filename, split_dir)
        for name, filename in input_files.items()
    }

    preprocessing_metadata = {
        "preprocessing": "original_features",
        "original_features": len(datasets["train"].columns) - 1,
        "pca_components": 0,
        "transformed_features": len(datasets["train"].columns) - 1,
        "pca_explained_variance_ratio_sum": np.nan,
    }
    feature_pipeline = config.get("feature_pipeline", "original")

    if feature_pipeline == "standard_scaler_pca_concat":
        print(
            "Ajustando StandardScaler y PCA exclusivamente con train "
            "y concatenando features estandarizadas + componentes..."
        )
        datasets, scaler, pca, preprocessing_metadata = (
            t5_feature_preprocessing.transformar_standard_scaler_pca(
                datasets=datasets,
                label_col=label_col,
                n_components=int(config["pca_components"]),
                random_state=config["random_state"],
            )
        )
        preprocessing_dir = model_dir / "preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, preprocessing_dir / "standard_scaler.joblib")
        joblib.dump(pca, preprocessing_dir / "pca.joblib")
    elif feature_pipeline == "robust_scaler_pca_concat":
        print(
            "Ajustando RobustScaler y PCA exclusivamente con train "
            "y concatenando features escaladas + componentes..."
        )
        datasets, scaler, pca, preprocessing_metadata = (
            t5_feature_preprocessing.transformar_robust_scaler_pca(
                datasets=datasets,
                label_col=label_col,
                n_components=int(config["pca_components"]),
                random_state=config["random_state"],
            )
        )
        preprocessing_dir = model_dir / "preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, preprocessing_dir / "robust_scaler.joblib")
        joblib.dump(pca, preprocessing_dir / "pca.joblib")
    elif feature_pipeline != "original":
        raise ValueError(f"Pipeline de features no reconocido: {feature_pipeline}")

    print("Transformando datasets tabulares a texto...")
    text_datasets = {
        name: t5_data.dataframe_a_source_target(df, label_col=label_col)
        for name, df in datasets.items()
    }

    labels = sorted(datasets["test"][label_col].astype(str).unique())
    model = T5Classifier().cargar_modelo(
        model_name_or_path=model_base_path,
        use_gpu=config["use_gpu"],
        require_gpu=config["require_gpu"],
    )

    print("Iniciando fine-tuning...")
    fit_start = time.time()
    trainer = model.entrenar(
        train_df=text_datasets["train"],
        val_df=text_datasets["validation"],
        source_max_token_len=config["source_max_token_len"],
        target_max_token_len=config["target_max_token_len"],
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        output_dir=checkpoint_dir,
        best_model_dir=best_model_dir,
        precision=config["precision"],
        use_gpu=config["use_gpu"],
        require_gpu=config["require_gpu"],
        dataloader_num_workers=config["dataloader_num_workers"],
        random_state=config["random_state"],
    )
    fit_time = time.time() - fit_start

    print("Evaluando el mejor modelo sobre test...")
    score_start = time.time()
    predictions = model.predecir(
        text_datasets["test"][t5_data.SOURCE_TEXT_COL].tolist(),
        batch_size=config["batch_size"],
        source_max_token_len=config["source_max_token_len"],
        target_max_token_len=config["target_max_token_len"],
    )
    score_time = time.time() - score_start
    y_test = text_datasets["test"][t5_data.TARGET_TEXT_COL].astype(str)

    test_metrics = metrics.calcular_metricas_desde_predicciones(
        y_true=y_test,
        y_pred=predictions,
        labels_globales=labels,
        score_time=score_time,
    )
    test_metrics["fit_time"] = fit_time

    parameters = {
        "dataset": dataset,
        "experiment": experiment_name,
        "phase": phase_label,
        "fold": fold,
        "model_base_path": str(model_base_path),
        "selection_metric": "accuracy",
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_accuracy": trainer.state.best_metric,
        "source_max_token_len": config["source_max_token_len"],
        "target_max_token_len": config["target_max_token_len"],
        "batch_size": config["batch_size"],
        "max_epochs": config["max_epochs"],
        "precision": config["precision"],
        "random_state": config["random_state"],
        "train_rows": len(datasets["train"]),
        "validation_rows": len(datasets["validation"]),
        "test_rows": len(datasets["test"]),
        "features": len(datasets["train"].columns) - 1,
        **preprocessing_metadata,
    }
    pd.DataFrame([parameters]).to_csv(
        report_dir / f"{dataset}__{experiment_name}__experiment_parameters.csv",
        index=False,
    )
    pd.DataFrame([{"dataset": dataset, "experiment": experiment_name, **test_metrics}]).to_csv(
        report_dir / f"{dataset}__{experiment_name}__test_metrics.csv",
        index=False,
    )

    prediction_report = pd.DataFrame(
        {
            "test_index": np.arange(len(y_test)),
            "true_label": y_test.to_numpy(),
            "predicted_label": predictions,
        }
    )
    prediction_report["is_valid_label"] = prediction_report["predicted_label"].isin(labels)
    prediction_report["is_correct"] = (
        prediction_report["true_label"] == prediction_report["predicted_label"]
    )
    prediction_report.to_csv(
        report_dir / f"{dataset}__{experiment_name}__test_predictions.csv",
        index=False,
    )

    for name, df in datasets.items():
        _guardar_distribucion_etiquetas(
            df,
            label_col,
            report_dir / f"{dataset}__{experiment_name}__{name}_label_distribution.csv",
        )

    _guardar_matriz_confusion(
        y_true=y_test.tolist(),
        y_pred=predictions,
        labels=labels,
        csv_path=report_dir / f"{dataset}__{experiment_name}__test_confusion_matrix.csv",
        figure_path=figure_dir / f"{dataset}__{experiment_name}__test_confusion_matrix.pdf",
    )
    _guardar_historial_entrenamiento(
        trainer=trainer,
        csv_path=report_dir / f"{dataset}__{experiment_name}__training_history.csv",
        figure_path=figure_dir / f"{dataset}__{experiment_name}__training_loss.pdf",
    )

    with (report_dir / f"{dataset}__{experiment_name}__trainer_state.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(asdict(trainer.state), file, indent=2, default=str)

    print("Experimento finalizado correctamente.")
    print(f"Mejor modelo: {best_model_dir}")
    print(f"Accuracy test: {test_metrics['accuracy']:.6f}")
    print(f"Reportes: {report_dir}")

    return test_metrics
