import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import evaluate
import numpy as np


class NerClassifier(ABC):
    """Base class for NER models fine-tuning.

    Attributes:
        label_names (list[str]): list of possible labels.

    Methods:
        train(self, model_name: str, train_data: Any, val_data: Any, **kwargs: Any) -> None:
            Trains the provided model.

        predict(self, test_data: list[list[str]]) -> list[list[str]]:
            Gets predictions for test data.

        compute_metrics(self, eval_preds: Union[list[np.ndarray], list[list[str]]]) -> dict:
            Calculates a set of metrics for previously predicted results.
    """

    def __init__(self, label_names: list[str]):
        """Initializes the NerClassifier with a list of possible labels."""
        self.metric = evaluate.load("seqeval")
        self.label_names = label_names
        # Mapping labels to numbers
        self.id2label = {i: label for i, label in enumerate(label_names)}
        self.label2id = {v: k for k, v in self.id2label.items()}

    @abstractmethod
    def train(self, model_name: str, train_data: Any, val_data: Any, **kwargs: Any):
        """Trains the provided model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, samples: list[str]):
        """Gets predictions for test data."""
        raise NotImplementedError

    def compute_metrics(self, eval_preds: Union[list[np.ndarray], list[list[str]]]):
        """Calculates a set of metrics for previously predicted results."""
        logits, labels = eval_preds
        
        predictions = np.argmax(logits, axis=-1)

        # Strip B-/I- prefixes to align evaluation (treat boundary tags as same class)
        def strip_prefix(tag: str) -> str:
            return tag[2:] if tag.startswith(("B-", "I-")) else tag

        true_labels = [
            [strip_prefix(self.id2label[l]) for l in label if l != -100]
            for label in labels
        ]
        
        true_predictions = [
            [strip_prefix(self.id2label[p]) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        all_metrics = self.metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    def save(self, path: str):
        """Saves model under path folder

        Args:
            path (str): folder to save the model
        """
        os.makedirs(Path(path), exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
        params = {
            "model_checkpoint": self.model_checkpoint,
            "label_names": self.label_names,
            "id2label": self.id2label,
            "label2id": self.label2id
        }

        with open(Path(path) / "model_conf.json", "w") as out:
            json.dump(params, out)