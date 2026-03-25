import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from pathlib import Path
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch



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


def undersample_clases_mayores(df, n_max, label_col="LABEL", random_state=42):
    """
    Aplica RandomUnderSampler a todas las clases que tengan más de n_max muestras.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset completo
    n_max : int
        Número máximo de muestras por clase
    label_col : str
        Nombre de la columna de etiquetas
    random_state : int
        Semilla para reproducibilidad

    Returns
    -------
    df_res : pandas.DataFrame
        Dataset balanceado
    """

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Conteo de clases
    class_counts = y.value_counts()

    # Construir sampling_strategy SOLO para clases que superen n_max
    sampling_strategy = {
        clase: n_max
        for clase, count in class_counts.items()
        if count > n_max
    }

    # Si ninguna clase supera n_max, devolvemos el df tal cual
    if len(sampling_strategy) == 0:
        return df.copy()

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



def balancear_nearmiss_y_mover_a_test(
    df_train,
    df_test,
    label_col="LABEL",
    target_n=10000,
    random_state=42,
    k_neighbors=5,
    version_nearmiss=1,
    n_neighbors_nearmiss=3
):
    df_train = df_train.copy()
    df_test = df_test.copy()

    # Separar X e y
    X_train = df_train.drop(columns=[label_col])
    y_train = df_train[label_col]

    class_counts = y_train.value_counts()

    # --------- UNDER con NearMiss ----------
    # Solo las clases con más de target_n se reducen a target_n
    under_strategy = {
        cls: target_n
        for cls, count in class_counts.items()
        if count > target_n
    }

    if under_strategy:
        nm = NearMiss(
            sampling_strategy=under_strategy,
            version=version_nearmiss,
            n_neighbors=n_neighbors_nearmiss
        )

        X_under, y_under = nm.fit_resample(X_train, y_train)

        # Índices seleccionados por NearMiss que se quedan en train
        indices_keep = df_train.index[nm.sample_indices_]

        # Índices eliminados que se moverán a test
        indices_remove = df_train.index.difference(indices_keep)

        # Mover eliminados a test
        df_test = pd.concat([df_test, df_train.loc[indices_remove]], ignore_index=True)

        # Reconstruir train tras NearMiss
        df_train = pd.DataFrame(X_under, columns=X_train.columns)
        df_train[label_col] = y_under
    else:
        indices_remove = []

    # --------- SMOTE ----------
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

        if k >= 1:
            smote = SMOTE(
                sampling_strategy=over_strategy,
                random_state=random_state,
                k_neighbors=k
            )

            X_res, y_res = smote.fit_resample(X, y)

            df_train = pd.DataFrame(X_res, columns=X.columns)
            df_train[label_col] = y_res

    return df_train, df_test

class SimpleT5Wrapper:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_type = None
        self.model_name_or_path = None
        self.device = torch.device("cpu")

    def from_pretrained(self, model_type, model_name_or_path, use_gpu=True):
        if model_type.lower() != "t5":
            raise ValueError("Solo se soporta model_type='t5'.")

        self.model_type = model_type
        self.model_name_or_path = str(model_name_or_path)

        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name_or_path)

        # mover explícitamente el modelo al device correcto
        self.model.to(self.device)

    def _preprocess_function(self, examples, source_max_token_len, target_max_token_len):
        model_inputs = self.tokenizer(
            examples["source_text"],
            max_length=source_max_token_len,
            truncation=True,
            padding=False
        )

        labels = self.tokenizer(
            examples["target_text"],
            max_length=target_max_token_len,
            truncation=True,
            padding=False
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _compute_metrics_builder(self):
        tokenizer = self.tokenizer

        def compute_metrics(eval_preds):
            preds, labels = eval_preds

            if isinstance(preds, tuple):
                preds = preds[0]

            if preds.ndim == 3:
                pred_ids = np.argmax(preds, axis=-1)
            else:
                pred_ids = preds

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            pred_texts = [p.strip() for p in pred_texts]
            label_texts = [l.strip() for l in label_texts]

            accuracy = sum(p == l for p, l in zip(pred_texts, label_texts)) / len(label_texts)
            return {"accuracy": accuracy}

        return compute_metrics

    def train(
        self,
        train_df,
        eval_df,
        source_max_token_len=512,
        target_max_token_len=128,
        batch_size=16,
        max_epochs=10,
        outputdir="outputs",
        precision=32,
        use_gpu=True,
        dataloader_num_workers=16
    ):
        if self.tokenizer is None or self.model is None:
            raise ValueError("Primero debes llamar a from_pretrained().")

        outputdir = str(outputdir)
        Path(outputdir).mkdir(parents=True, exist_ok=True)

        train_df = train_df.copy()
        eval_df = eval_df.copy()

        train_df["source_text"] = train_df["source_text"].astype(str)
        train_df["target_text"] = train_df["target_text"].astype(str)

        eval_df["source_text"] = eval_df["source_text"].astype(str)
        eval_df["target_text"] = eval_df["target_text"].astype(str)

        train_dataset = Dataset.from_pandas(
            train_df[["source_text", "target_text"]],
            preserve_index=False
        )
        eval_dataset = Dataset.from_pandas(
            eval_df[["source_text", "target_text"]],
            preserve_index=False
        )

        preprocess = lambda examples: self._preprocess_function(
            examples,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len
        )

        train_dataset = train_dataset.map(preprocess, batched=True)
        eval_dataset = eval_dataset.map(preprocess, batched=True)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )

        fp16 = False
        if precision == 16:
            fp16 = True
        elif precision != 32:
            raise ValueError("precision solo puede ser 16 o 32.")

        # reforzar device del modelo
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        training_args = Seq2SeqTrainingArguments(
            output_dir=outputdir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=max_epochs,
            predict_with_generate=True,
            generation_max_length=target_max_token_len,
            dataloader_num_workers=dataloader_num_workers,
            fp16=fp16,
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none"
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics_builder()
        )

        trainer.train()
        trainer.save_model(outputdir)
        self.tokenizer.save_pretrained(outputdir)

        return trainer