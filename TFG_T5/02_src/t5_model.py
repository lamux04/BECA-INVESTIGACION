"""Modelo T5 para clasificacion generativa de datos tabulares."""

from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

import t5_data


class T5Classifier:
    """Encapsula la carga, entrenamiento y prediccion de un modelo T5."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_name_or_path = None
        self.device = torch.device("cpu")

    def cargar_modelo(
        self,
        model_name_or_path,
        use_gpu=True,
        require_gpu=False,
    ):
        """Carga tokenizer y pesos desde Hugging Face o una ruta local."""
        self.model_name_or_path = str(model_name_or_path)
        self.device = self._seleccionar_device(use_gpu, require_gpu=require_gpu)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name_or_path
        )
        self.model.to(self.device)

        return self

    @staticmethod
    def _seleccionar_device(use_gpu, require_gpu=False):
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "Se solicito GPU, pero CUDA no esta disponible. "
                f"PyTorch={torch.__version__}, "
                f"CUDA de PyTorch={torch.version.cuda}. "
                "Comprueba que la build de PyTorch sea compatible con el "
                "driver NVIDIA del nodo."
            )

        return torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

    def _validar_modelo_cargado(self):
        if self.tokenizer is None or self.model is None:
            raise ValueError("Primero debes llamar a cargar_modelo().")

    def _tokenizar(self, examples, source_max_token_len, target_max_token_len):
        model_inputs = self.tokenizer(
            examples[t5_data.SOURCE_TEXT_COL],
            max_length=source_max_token_len,
            truncation=True,
            padding=False,
        )
        labels = self.tokenizer(
            examples[t5_data.TARGET_TEXT_COL],
            max_length=target_max_token_len,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def _construir_compute_metrics(self):
        tokenizer = self.tokenizer

        def compute_metrics(eval_preds):
            predictions, labels = eval_preds

            if isinstance(predictions, tuple):
                predictions = predictions[0]

            prediction_ids = (
                np.argmax(predictions, axis=-1)
                if predictions.ndim == 3
                else predictions
            )
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            prediction_texts = tokenizer.batch_decode(
                prediction_ids,
                skip_special_tokens=True,
            )
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            prediction_texts = [text.strip() for text in prediction_texts]
            label_texts = [text.strip() for text in label_texts]
            correct = sum(
                prediction == label
                for prediction, label in zip(prediction_texts, label_texts)
            )

            return {"accuracy": correct / len(label_texts) if label_texts else 0.0}

        return compute_metrics

    def entrenar(
        self,
        train_df,
        val_df,
        source_max_token_len=512,
        target_max_token_len=128,
        batch_size=16,
        max_epochs=10,
        output_dir="outputs",
        best_model_dir=None,
        precision=32,
        use_gpu=True,
        require_gpu=False,
        dataloader_num_workers=16,
        random_state=42,
    ):
        """Entrena el modelo y guarda el mejor checkpoint en ``output_dir``."""
        self._validar_modelo_cargado()
        t5_data.validar_source_target(train_df, "train")
        t5_data.validar_source_target(val_df, "validation")

        if precision not in {16, 32}:
            raise ValueError("precision solo puede ser 16 o 32.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model_dir = Path(best_model_dir or output_dir / "best_model")
        best_model_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = self._crear_dataset_huggingface(train_df)
        val_dataset = self._crear_dataset_huggingface(val_df)

        def preprocess(examples):
            return self._tokenizar(
                examples,
                source_max_token_len=source_max_token_len,
                target_max_token_len=target_max_token_len,
            )

        train_dataset = train_dataset.map(preprocess, batched=True)
        val_dataset = val_dataset.map(preprocess, batched=True)

        self.device = self._seleccionar_device(
            use_gpu,
            require_gpu=require_gpu,
        )
        self.model.to(self.device)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=max_epochs,
            predict_with_generate=True,
            generation_max_length=target_max_token_len,
            dataloader_num_workers=dataloader_num_workers,
            fp16=precision == 16,
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none",
            seed=random_state,
            data_seed=random_state,
        )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
            ),
            compute_metrics=self._construir_compute_metrics(),
        )

        trainer.train()
        trainer.save_model(str(best_model_dir))
        self.tokenizer.save_pretrained(str(best_model_dir))

        return trainer

    @staticmethod
    def _crear_dataset_huggingface(df):
        data = df[[t5_data.SOURCE_TEXT_COL, t5_data.TARGET_TEXT_COL]].copy()
        data[t5_data.SOURCE_TEXT_COL] = data[t5_data.SOURCE_TEXT_COL].astype(str)
        data[t5_data.TARGET_TEXT_COL] = data[t5_data.TARGET_TEXT_COL].astype(str)

        return Dataset.from_pandas(data, preserve_index=False)

    def predecir(
        self,
        texts,
        batch_size=16,
        source_max_token_len=150,
        target_max_token_len=3,
    ):
        """Genera una etiqueta textual para cada entrada."""
        self._validar_modelo_cargado()

        if isinstance(texts, str):
            texts = [texts]

        texts = list(texts)
        self.model.eval()
        predictions = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=source_max_token_len,
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=target_max_token_len,
                    num_beams=1,
                )

            decoded = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )
            predictions.extend(text.strip() for text in decoded)

        return predictions
