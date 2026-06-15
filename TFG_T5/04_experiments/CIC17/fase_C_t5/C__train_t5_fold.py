"""Entrena y evalua T5 con features escaladas y PCA para un fold de CIC17."""

# ============== LIBRERIAS =============
import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "02_src"
sys.path.append(str(SRC_DIR))

import t5_experiment_runner


# ============== PARAMETROS =============
DATASET_ORIGINAL = "CIC17"
LABEL_COL = "LABEL"
PCA_COMPONENTS = 33
SOURCE_MAX_TOKEN_LEN = 150
TARGET_MAX_TOKEN_LEN = 3
BATCH_SIZE = 16
MAX_EPOCHS = 10
PRECISION = 32
DATALOADER_NUM_WORKERS = 16
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "07_models" / "pretrained" / "t5-small",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Permite continuar en CPU si CUDA no esta disponible.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_gpu = not args.cpu

    config = {
        "project_root": PROJECT_ROOT,
        "dataset": DATASET_ORIGINAL,
        "fold": args.fold,
        "label_col": LABEL_COL,
        "phase_dir_name": "fase_C_t5",
        "phase_label": "C_t5",
        "experiment_name_template": "C__t5_fold_{fold}",
        "feature_pipeline": "standard_scaler_pca_concat",
        "pca_components": PCA_COMPONENTS,
        "model_base_path": args.model_path,
        "source_max_token_len": SOURCE_MAX_TOKEN_LEN,
        "target_max_token_len": TARGET_MAX_TOKEN_LEN,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "precision": PRECISION,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "random_state": RANDOM_STATE,
        "use_gpu": use_gpu,
        "require_gpu": use_gpu and not args.allow_cpu_fallback,
    }
    t5_experiment_runner.ejecutar_experimento_t5(config)


if __name__ == "__main__":
    main()
