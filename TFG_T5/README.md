# TFG

Repositorio del TFG centrado en la deteccion de ciberataques con T5.

## Pipeline CIC17

1. `03_preprocessing/CIC17/01_clean_dataset.py`: limpieza del dataset.
2. `03_preprocessing/CIC17/02_split_dataset.py`: train/test y cinco folds train/validation.
3. `04_experiments/CIC17/fase_A1_pca_components`: analisis PCA conservado como referencia.
4. `04_experiments/CIC17/fase_B_t5/B__train_t5_fold.py`: fine-tuning y evaluacion de un fold.

El modelo base `t5-small` debe estar en `07_models/pretrained/t5-small/`.
Cada job de `06_jobs/bash/CIC17/fase_B_t5/` entrena un fold con GPU y guarda el
mejor modelo segun accuracy de validacion.

## PyTorch en el cluster

Los nodos GPU actuales usan un driver compatible con CUDA 12.6. El entorno
`tfgClean` debe utilizar la build CUDA 12.6 de PyTorch:

```bash
conda activate tfgClean
python -m pip install --force-reinstall \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

Los jobs de fase B comprueban `torch.cuda.is_available()` antes de comenzar el
entrenamiento y muestran las versiones detectadas en el log.
