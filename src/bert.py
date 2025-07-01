# src/bert.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Union, Dict

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertTokenizerFast,
    LayoutLMv3TokenizerFast,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from utils.bert import tokenize_and_align_labels  # updated version with windows
from utils.label_aggregation import PredictionAggregator
from utils.postprocess import snap_boundaries

from .base import NerClassifier
from .schema import BERTDataset

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
#else MPS not supported reliably for CRF; fall back to CPU
#elseif torch.backends.mps.is_available():
#    device = torch.device("mps")
else:
    device = torch.device("cpu")

# -----------------------------------------------------------------------------
# Focal loss for token classification
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=alpha, ignore_index=ignore_index, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)  # (N,)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()

# -----------------------------------------------------------------------------
# Weighted trainer using FocalLoss
# -----------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, gamma: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = FocalLoss(class_weights, gamma=gamma, ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, L, C)
        active = labels.view(-1) != -100
        active_logits = logits.view(-1, model.config.num_labels)[active]
        active_labels = labels.view(-1)[active]
        loss = self.loss_fct(active_logits, active_labels)
        return (loss, outputs) if return_outputs else loss

# -----------------------------------------------------------------------------
# Simple trainer with weighted Cross-Entropy (no focal loss)
# -----------------------------------------------------------------------------
class CETrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, L, C)

        # ------------------------------------------------------------------
        # If the model contains a CRF layer, use negative log-likelihood from CRF
        # Otherwise fall back to (weighted) Cross-Entropy as before.
        # ------------------------------------------------------------------
        if hasattr(model, "crf") and model.crf is not None:
            # Create mask from attention_mask (0 -> padding) and make sure it is bool
            mask = inputs.get("attention_mask").bool()
            # CRF requires labels with no ignore_index; replace -100 with a valid tag (e.g. 0)
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            # Negative log-likelihood; model.crf returns log-likelihood, hence the minus sign
            loss = -model.crf(logits, labels_crf, mask=mask, reduction="mean")
        else:
            active = labels.view(-1) != -100
            loss = self.loss_fct(
                logits.view(-1, model.config.num_labels)[active],
                labels.view(-1)[active],
            )
        return (loss, outputs) if return_outputs else loss

# -----------------------------------------------------------------------------
# Helper: BIO‑aware span merger
# -----------------------------------------------------------------------------
class _SpanMerger:
    def __init__(self, id2label: Dict[int, str], label2id: Dict[str, int], threshold: float = 0.7):
        self.id2label = id2label
        self.label2id = label2id
        self.threshold = threshold
        # Group indices
        self.ant_ids = [label2id["B-Antagonist"], label2id["I-Antagonist"]]
        self.prot_ids = [label2id["B-Protagonist"], label2id["I-Protagonist"]]
        self.inn_ids = [label2id["B-Innocent"], label2id["I-Innocent"]]

        if "B-Unknown" in label2id:
            self.unk_ids = [label2id["B-Unknown"], label2id["I-Unknown"]]
        else:
            self.unk_ids = []

    def merge(self, raw_text: str, preds: np.ndarray, probs: np.ndarray, offsets: np.ndarray):
        """Return list of entity dicts."""
        spans = []
        cur = None  # current span dict
        
        for i, lid in enumerate(preds):
            # Skip invalid offsets (special tokens, padding, etc.)
            start_pos, end_pos = offsets[i]
            if start_pos == end_pos or start_pos == 0 and end_pos == 0:
                # Flush current span when encountering invalid token
                self._flush(cur, spans, raw_text)
                cur = None
                continue
                
            label = self.id2label[lid]
            
            if label.startswith("B-"):
                # If a new B-tag follows immediately after the current span *of the same class*,
                # treat it as a continuation (handles sequences like B-X B-X B-X → one span).
                if cur is not None and cur["type"] == label[2:] and self._is_adjacent(cur["end"], start_pos):
                    cur["end"] = end_pos
                    cur["probs"].append(probs[i])
                else:
                    self._flush(cur, spans, raw_text)
                    cur = {"type": label[2:], "start": start_pos, "end": end_pos, "probs": [probs[i]]}
            elif label.startswith("I-") and cur is not None and cur["type"] == label[2:]:
                cur["end"] = end_pos
                cur["probs"].append(probs[i])
            else:
                self._flush(cur, spans, raw_text)
                cur = None
                
        self._flush(cur, spans, raw_text)
        
        # Post-process: merge nearby spans of the same type
        spans = self._merge_nearby_spans(spans, raw_text)
        return spans

    def _merge_nearby_spans(self, spans, raw_text):
        """Merge spans of the same type that are very close to each other."""
        if len(spans) <= 1:
            return spans
        
        merged = []
        i = 0
        while i < len(spans):
            current = spans[i]
            
            # Look ahead for spans of the same type that are close
            j = i + 1
            while j < len(spans):
                next_span = spans[j]
                
                # Check if same type and close enough
                same_type = (
                    current["prob_antagonist"] == next_span["prob_antagonist"] and
                    current["prob_protagonist"] == next_span["prob_protagonist"] and  
                    current["prob_innocent"] == next_span["prob_innocent"]
                )
                
                # Get the best role for both spans
                current_best = max([
                    (current["prob_antagonist"], "Antagonist"),
                    (current["prob_protagonist"], "Protagonist"),
                    (current["prob_innocent"], "Innocent"),
                    (current["prob_unknown"], "Unknown")
                ])[1]
                
                next_best = max([
                    (next_span["prob_antagonist"], "Antagonist"),
                    (next_span["prob_protagonist"], "Protagonist"),
                    (next_span["prob_innocent"], "Innocent"),
                    (next_span["prob_unknown"], "Unknown")
                ])[1]
                
                gap = next_span["start"] - current["end"]
                if current_best == next_best and 0 <= gap <= 5:  # Same type and close
                    # Merge spans
                    current["end"] = next_span["end"]
                    current["word"] = raw_text[current["start"]:current["end"]]
                    
                    # Average the probabilities
                    current["prob_antagonist"] = (current["prob_antagonist"] + next_span["prob_antagonist"]) / 2
                    current["prob_protagonist"] = (current["prob_protagonist"] + next_span["prob_protagonist"]) / 2
                    current["prob_innocent"] = (current["prob_innocent"] + next_span["prob_innocent"]) / 2
                    current["prob_unknown"] = (current["prob_unknown"] + next_span["prob_unknown"]) / 2
                    
                    j += 1  # Skip the merged span
                else:
                    break  # No more spans to merge
            
            merged.append(current)
            i = j if j > i + 1 else i + 1  # Move to next unprocessed span
        
        return merged

    # ------------------------------------------------------------------
    def _validate_entity(self, span: Dict) -> bool:
        """Validate if a span represents a valid entity."""
        return True

    def _flush(self, cur: Dict | None, spans: List[Dict], raw_text: str):
        if cur is None:
            return
        token_probs = np.mean(cur["probs"], axis=0)
        raw_ant = float(token_probs[self.ant_ids].sum())
        raw_prot = float(token_probs[self.prot_ids].sum())
        raw_inn = float(token_probs[self.inn_ids].sum())
        raw_o = float(token_probs[self.label2id["O"]])

        if self.unk_ids:
            raw_unk = float(token_probs[self.unk_ids].sum())
        else:
            raw_unk = 0.0

        raw_max = max(raw_ant, raw_prot, raw_inn, raw_unk)
        if raw_max > raw_o and raw_max > self.threshold:
            total_non_o = raw_ant + raw_prot + raw_inn
            ant = raw_ant / total_non_o if total_non_o else 0.0
            prot = raw_prot / total_non_o if total_non_o else 0.0
            inn = raw_inn / total_non_o if total_non_o else 0.0

            if self.unk_ids:
                unk = raw_unk / total_non_o if total_non_o else 0.0
            else:
                unk = 0.0
            
            span = {
                "start": cur["start"],
                "end": cur["end"],
                "word": raw_text[cur["start"]:cur["end"]],
                "prob_antagonist": ant,
                "prob_protagonist": prot,
                "prob_innocent": inn,
                "prob_unknown": unk,
            }
            
            # Snap span boundaries (fix off-by-one / punctuation issues)
            span = snap_boundaries(span, raw_text)
            
            # Only add span if it passes validation
            if self._validate_entity(span):
                spans.append(span)

    # ------------------------------------------------------------------
    @staticmethod
    def _is_adjacent(prev_end: int, next_start: int) -> bool:
        """Return True if two tokens are adjacent in the original text (allowing 1-char gap)."""
        return 0 <= next_start - prev_end <= 3  # Allow up to 3 characters gap (for spaces, punctuation)

# -----------------------------------------------------------------------------
# Main classifier
# -----------------------------------------------------------------------------
class BertNerClassifier(NerClassifier):
    """BERT‑based (or similar) token‑classification model for NER."""

    MAX_LENGTH = 512
    DOC_STRIDE = 128

    # ------------------------------------------------------------------
    def __init__(self, label_names: List[str], model_checkpoint: str = "bert-base-uncased", **kwargs):
        super().__init__(label_names)
        self.model_checkpoint = model_checkpoint
        self.max_length = kwargs.get("max_length", self.MAX_LENGTH)
        self.doc_stride = kwargs.get("doc_stride", self.DOC_STRIDE)

        # Try to initialize the fast tokenizer, fall back to slow if it fails (e.g., missing protobuf)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        except (ValueError, ImportError) as e:
            print(f"⚠️ Warning: fast tokenizer init failed ({e}), falling back to slow tokenizer.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=len(label_names),
            id2label=self.id2label,
            label2id=self.label2id,
        ).to(device)

        self.prediction_aggregator = PredictionAggregator(self)
        self.merger = _SpanMerger(self.id2label, self.label2id, threshold=0.5)

    # ------------------------------------------------------------------
    def _tokenize_and_align(self, dataset: BERTDataset) -> Dataset:
        return tokenize_and_align_labels(
            dataset,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            doc_stride=self.doc_stride,
        )

    # ------------------------------------------------------------------
    def train(self, model_name: str, train_data: BERTDataset, val_data: BERTDataset, **kwargs: Any):
        # ----- tokenise
        train_dataset = self._tokenize_and_align(train_data)
        val_dataset = self._tokenize_and_align(val_data)

        # ----- improved class-weights calculation
        # count how many *tokens* of each label appear in the training set
        flat_labels = [l for seq in train_dataset["labels"] for l in seq if l != -100]
        label_counts = np.bincount(flat_labels, minlength=len(self.label_names))
        
        # Print label distribution for debugging
        print("Label distribution in training data:")
        for i, (label, count) in enumerate(zip(self.label_names, label_counts)):
            print(f"  {label}: {count} tokens ({count/len(flat_labels)*100:.1f}%)")

        # More conservative class weights to prevent over-weighting rare classes
        total_tokens = len(flat_labels)
        class_weights = []
        for count in label_counts:
            if count == 0:
                weight = 1.0
            else:
                # Inverse frequency for more balanced weighting
                weight = total_tokens / count
            class_weights.append(weight)
        
        # Keep O class at weight 1 and scale others relative to it
        o_weight = class_weights[0] if label_counts[0] > 0 else 1.0
        class_weights = [w / o_weight for w in class_weights]
        class_weights[0] = 1.0

        # Scale down weights to avoid extreme gradients (overrideable per-model)
        scale = kwargs.pop("class_weight_scale", 0.6)
        class_weights = [w if i == 0 else w * scale for i, w in enumerate(class_weights)]

        # Optionally down-weight the *Unknown* label(s) further to mitigate over-generation
        unknown_scale = kwargs.pop("unknown_weight_scale", 1.0)
        if unknown_scale != 1.0:
            for idx, lbl in enumerate(self.label_names):
                if "Unknown" in lbl:
                    class_weights[idx] *= unknown_scale

        print("Applied class weights:")
        for label, weight in zip(self.label_names, class_weights):
            print(f"  {label}: {weight:.2f}")

        class_wts = torch.tensor(class_weights, dtype=torch.float32, device=device)

        # ----- improved training args
        training_args = TrainingArguments(
            output_dir=model_name,
            learning_rate=kwargs.get("learning_rate", 5e-5),  # Slightly higher LR
            num_train_epochs=kwargs.get("num_train_epochs", 5),
            weight_decay=kwargs.get("weight_decay", 0.01),
            per_device_train_batch_size=kwargs.get("batch_size", 16),  # Larger batch size
            per_device_eval_batch_size=kwargs.get("batch_size", 16),
            lr_scheduler_type="cosine",  # Better scheduler
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            gradient_accumulation_steps=kwargs.get("grad_accum_steps", 1),
            eval_strategy="epoch",  # Evaluate at end of each epoch
            save_strategy="epoch",  # Save at end of each epoch
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
            logging_strategy="epoch",
            save_total_limit=3,  # Keep only best 3 checkpoints
        )

        # ----- choose trainer (CE vs Focal)
        use_focal = kwargs.pop("use_focal", False)
        gamma = kwargs.pop("focal_gamma", 2.0)

        TrainerCls = WeightedTrainer if use_focal else CETrainer

        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": val_dataset,
            "data_collator": self.data_collator,
            "compute_metrics": self.compute_metrics,
            "callbacks": [EarlyStoppingCallback(early_stopping_patience=5)],
            "class_weights": class_wts,
        }

        if use_focal:
            trainer_kwargs["gamma"] = gamma

        self.trainer = TrainerCls(**trainer_kwargs)
        self.trainer.train()

    # ------------------------------------------------------------------
    # Inference helpers (windowed)
    # ------------------------------------------------------------------
    def _window_encode(self, raw_text: str):
        return self.tokenizer(
            raw_text,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_overflowing_tokens=True,
            stride=self.doc_stride,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',  # pad all windows to max_length for uniform tensor shapes
            return_tensors="pt",
        )

    def _forward_windows(self, encodings):
        all_preds, all_probs, all_offsets = [], [], []
        for i in range(encodings["input_ids"].size(0)):
            inputs = {k: v[i].unsqueeze(0).to(device) for k, v in encodings.items() if k in ["input_ids", "attention_mask"]}
            with torch.no_grad():
                logits = self.model(**inputs).logits  # (1, L, C)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            preds = probs.argmax(axis=-1)
            all_preds.append(preds)
            all_probs.append(probs)
            all_offsets.append(encodings["offset_mapping"][i].cpu().numpy())
        return all_preds, all_probs, all_offsets

    # ------------------------------------------------------------------
    def _predict_raw_text(self, raw_text: str):
        enc = self._window_encode(raw_text)
        preds, probs, offsets = self._forward_windows(enc)
        # flatten windows
        preds = np.concatenate(preds, axis=0)
        probs = np.concatenate(probs, axis=0)
        offsets = np.concatenate(offsets, axis=0)
        return self.merger.merge(raw_text, preds, probs, offsets)

    def _predict_raw_text_tokens(self, raw_text: str):
        enc = self._window_encode(raw_text)
        preds, _, offsets = self._forward_windows(enc)
        preds = np.concatenate(preds, axis=0)
        offsets = np.concatenate(offsets, axis=0)
        special_mask = (offsets[:, 0] == offsets[:, 1])
        token_predictions = []
        for lid, (s, e), is_special in zip(preds, offsets, enc["special_tokens_mask"][0].cpu().numpy()):
            if is_special or s == e:
                continue
            token_predictions.append({
                "start": int(s),
                "end": int(e),
                "entity": self.id2label[int(lid)],
                "word": raw_text[int(s):int(e)],
            })
        return token_predictions

    # ------------------------------------------------------------------
    def _predict_token_list_bio(self, tokens: List[str]):
        # rebuild text + map positions
        raw_text, positions = "", []
        cur = 0
        for i, tok in enumerate(tokens):
            if i > 0:
                raw_text += " "
                cur += 1
            start = cur
            raw_text += tok
            cur += len(tok)
            positions.append((start, cur))
        token_preds = self._predict_raw_text_tokens(raw_text)
        pred_map = {p["start"]: p["entity"] for p in token_preds}
        return [{"start": s, "end": e, "entity": pred_map.get(s, "O"), "word": tokens[i]} for i, (s, e) in enumerate(positions)]

    # ------------------------------------------------------------------
    def predict(self, samples, aggregation_strategy="first", return_format="spans"):
        if isinstance(samples, str):
            return self._predict_raw_text(samples) if return_format == "spans" else self._predict_raw_text_tokens(samples)
        if isinstance(samples, dict) and "tokens" in samples:
            if return_format == "spans":
                return self._predict_raw_text(" ".join(samples["tokens"][0]))
            return self._predict_token_list_bio(samples["tokens"][0])
        raise ValueError(f"Unsupported samples type for predict: {type(samples)}")

    # ------------------------------------------------------------------
    @staticmethod
    def load(path: str) -> "BertNerClassifier":
        with open(Path(path) / "model_conf.json", "r", encoding="utf-8") as f:
            params = json.load(f)
        inst = BertNerClassifier(params["label_names"], model_checkpoint=params["model_checkpoint"])
        inst.model = AutoModelForTokenClassification.from_pretrained(path).to(device)
        inst.tokenizer = AutoTokenizer.from_pretrained(path)
        inst.id2label = {int(k): v for k, v in params["id2label"].items()}
        inst.label2id = params["label2id"]
        inst.prediction_aggregator = PredictionAggregator(inst)
        inst.merger = _SpanMerger(inst.id2label, inst.label2id, threshold=0.5)
        return inst
