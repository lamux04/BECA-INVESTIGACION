"""Agrega las metricas de test obtenidas en los cinco folds de la fase D."""

from pathlib import Path
import re
import sys

import pandas as pd


DATASET_ORIGINAL = "CIC17"
N_FOLDS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUTA_RESULTADOS_FASE = (
    PROJECT_ROOT / "05_results" / DATASET_ORIGINAL / "fase_D_t5"
)
RUTA_SALIDA_COMPARACION = RUTA_RESULTADOS_FASE / "_comparison"

METRICAS_TEST = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "fpr_weighted",
    "fnr_weighted",
    "fpr_macro",
    "fnr_macro",
    "mcc",
    "detection_latency_seconds",
    "fit_time",
    "score_time",
]

FOLD_RE = re.compile(r"D__t5_fold_(\d+)")


def _extraer_fold(ruta_reporte):
    match = FOLD_RE.fullmatch(ruta_reporte.parents[1].name)
    if match is None:
        raise ValueError(f"No se pudo extraer el fold desde: {ruta_reporte}")
    return int(match.group(1))


def _localizar_reportes_metricas():
    rutas = sorted(
        RUTA_RESULTADOS_FASE.glob(
            "D__t5_fold_*/reports/*__test_metrics.csv"
        )
    )
    reportes_por_fold = {}

    for ruta in rutas:
        fold = _extraer_fold(ruta)
        if fold in reportes_por_fold:
            raise ValueError(
                f"Se encontro mas de un reporte de metricas para el fold {fold}."
            )
        reportes_por_fold[fold] = ruta

    folds_esperados = set(range(1, N_FOLDS + 1))
    folds_encontrados = set(reportes_por_fold)
    folds_faltantes = sorted(folds_esperados - folds_encontrados)
    folds_inesperados = sorted(folds_encontrados - folds_esperados)

    if folds_faltantes:
        raise FileNotFoundError(
            "Faltan resultados de los siguientes folds: "
            + ", ".join(map(str, folds_faltantes))
        )
    if folds_inesperados:
        raise ValueError(
            "Se encontraron folds fuera del rango esperado: "
            + ", ".join(map(str, folds_inesperados))
        )
    return reportes_por_fold


def _cargar_resultados_folds(reportes_por_fold):
    filas = []
    for fold, ruta in sorted(reportes_por_fold.items()):
        df_reporte = pd.read_csv(ruta)
        if len(df_reporte) != 1:
            raise ValueError(
                f"El reporte del fold {fold} debe contener exactamente una fila."
            )

        metricas_faltantes = [
            metrica for metrica in METRICAS_TEST if metrica not in df_reporte.columns
        ]
        if metricas_faltantes:
            raise KeyError(
                f"Faltan metricas en el fold {fold}: "
                + ", ".join(metricas_faltantes)
            )

        fila = {"dataset": DATASET_ORIGINAL, "phase": "D_t5", "fold": fold}
        fila.update(df_reporte.iloc[0][METRICAS_TEST].to_dict())
        filas.append(fila)

    return pd.DataFrame(filas).sort_values("fold").reset_index(drop=True)


def _calcular_resumen_metricas(df_folds):
    resumen = {
        "dataset": DATASET_ORIGINAL,
        "phase": "D_t5",
        "n_folds": len(df_folds),
    }
    for metrica in METRICAS_TEST:
        valores = pd.to_numeric(df_folds[metrica], errors="raise")
        resumen[f"{metrica}_mean"] = float(valores.mean())
        resumen[f"{metrica}_std"] = float(valores.std(ddof=1))
    return pd.DataFrame([resumen])


def main():
    reportes_por_fold = _localizar_reportes_metricas()
    df_folds = _cargar_resultados_folds(reportes_por_fold)
    df_resumen = _calcular_resumen_metricas(df_folds)

    RUTA_SALIDA_COMPARACION.mkdir(parents=True, exist_ok=True)
    ruta_folds = (
        RUTA_SALIDA_COMPARACION
        / f"{DATASET_ORIGINAL}__D_t5__test_metrics_by_fold.csv"
    )
    ruta_resumen = (
        RUTA_SALIDA_COMPARACION
        / f"{DATASET_ORIGINAL}__D_t5__test_metrics_summary.csv"
    )
    df_folds.to_csv(ruta_folds, index=False)
    df_resumen.to_csv(ruta_resumen, index=False)

    print("Comparativa de la fase D generada correctamente.")
    print(f"Folds agregados: {len(df_folds)}")
    print(f"Resultados por fold: {ruta_folds}")
    print(f"Media y desviacion tipica: {ruta_resumen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
