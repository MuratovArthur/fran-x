# src/deberta.py

from typing import List, Any
from torchcrf import CRF  # type: ignore
from .bert import BertNerClassifier, _SpanMerger, BERTDataset, device
from utils.postprocess import snap_boundaries
import numpy as np
import torch
from pathlib import Path
import unicodedata

class DebertaV3NerClassifier(BertNerClassifier):
    """DeBERTa v3-based token-classification model for NER."""

    def __init__(self, label_names: List[str], model_checkpoint: str = "microsoft/deberta-v3-base", **kwargs: Any):
        # Increase context window unless caller overrides
        kwargs.setdefault("max_length", 1024)
        kwargs.setdefault("doc_stride", 256)

        # Allow caller to tweak non-O bias; pop so it isn't forwarded to parent
        bias_val = kwargs.pop("non_o_bias", 0.2)

        super().__init__(label_names=label_names, model_checkpoint=model_checkpoint, **kwargs)

        # Attach a small CRF layer on top of the token-classification head
        num_labels = len(label_names)
        self.model.crf = CRF(num_tags=num_labels, batch_first=True).to(device)

        # ------------------------------------------------------------------
        # Add per-label bias to encourage non-O classes (simple class weighting)
        # ------------------------------------------------------------------
        if bias_val:
            bias_vector = torch.zeros(num_labels, device=self.model.classifier.bias.device)
            for lbl, idx in self.label2id.items():
                if lbl != "O":
                    bias_vector[idx] = float(bias_val)
            with torch.no_grad():
                self.model.classifier.bias += bias_vector

        # ------------------------------------------------------------------
        # Custom span-merger with per-label probability scaling
        # ------------------------------------------------------------------
        # instantiate scaled merger (threshold can still be overridden later)
        self.merger = ScaledSpanMerger(self.id2label, self.label2id, threshold=0.5)

    def train(self, model_name: str, train_data: BERTDataset, val_data: BERTDataset, **kwargs: Any):
        kwargs.setdefault("learning_rate", 1e-5)
        kwargs.setdefault("class_weight_scale", 2.0)
        kwargs.setdefault("unknown_weight_scale", 0.5)  # down-weight Unknown
        kwargs.setdefault("batch_size", 2)
        super().train(model_name, train_data, val_data, **kwargs)

    @staticmethod
    def load(path: str) -> "DebertaV3NerClassifier":
        """Load a fine-tuned DeBERTa-CRF model from local path or Hugging Face Hub.

        This method supports both local model paths and Hugging Face model identifiers.
        For HF models, it will automatically download and cache the model.
        """

        import json
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
        from utils.label_aggregation import PredictionAggregator

        # Check if this is a Hugging Face model identifier or local path
        is_hf_model = not Path(path).exists() and "/" in path
        
        if is_hf_model:
            # Handle Hugging Face model
            print(f"Loading model from Hugging Face Hub: {path}")
            
            # For HF models, we need to get the config first to determine label names
            config = AutoConfig.from_pretrained(path)
            label_names = list(config.label2id.keys())
            
            # Create un-initialised instance
            inst: "DebertaV3NerClassifier" = object.__new__(DebertaV3NerClassifier)
            
            # Set attributes
            inst.label_names = label_names
            inst.id2label = config.id2label
            inst.label2id = config.label2id
            inst.model_checkpoint = path
            inst.max_length = 1024
            inst.doc_stride = 256
            
            # Load tokenizer and model directly from HF Hub
            inst.tokenizer = AutoTokenizer.from_pretrained(path)
            inst.merger = ScaledSpanMerger(inst.id2label, inst.label2id, threshold=0.5)
            
            # Determine device
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            else:
                dev = torch.device("cpu")
            
            # Load the model from HF Hub
            inst.model = AutoModelForTokenClassification.from_pretrained(path).to(dev)
            
            # Add CRF layer if not present (for compatibility)
            if not hasattr(inst.model, 'crf'):
                from torchcrf import CRF
                inst.model.crf = CRF(num_tags=len(label_names), batch_first=True).to(dev)
            
            inst.prediction_aggregator = PredictionAggregator(inst)
            return inst
        
        else:
            # Handle local model (original logic)
            # ------------------------------
            # Basic metadata
            # ------------------------------
            meta_path = Path(path) / "model_conf.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                params = json.load(f)

            label_names = params["label_names"]

            # ------------------------------
            # Create *un-initialised* instance (skip __init__)
            # ------------------------------
            inst: "DebertaV3NerClassifier" = object.__new__(DebertaV3NerClassifier)

            # Manually set simple attributes that __init__ would normally set
            inst.label_names = label_names
            inst.id2label = {int(k): v for k, v in params["id2label"].items()}
            inst.label2id = params["label2id"]
            inst.model_checkpoint = params["model_checkpoint"]
            # reasonable defaults (same as __init__ fallback for DeBERTa)
            inst.max_length = 1024
            inst.doc_stride = 256

            # ------------------------------
            # Tokeniser and merger
            # ------------------------------
            inst.tokenizer = AutoTokenizer.from_pretrained(path)
            inst.merger = ScaledSpanMerger(inst.id2label, inst.label2id, threshold=0.5)

            # ------------------------------
            # Load backbone (fine-tuned) once
            # ------------------------------
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            else:
                dev = torch.device("cpu")

            # -------------------------------------------------------------
            # Load backbone weights **excluding** CRF parameters so the HF
            # loader does not complain about unexpected keys.
            # -------------------------------------------------------------
            # Locate state-dict file (supports .safetensors or .bin)
            state_path_bin = Path(path) / "pytorch_model.bin"
            state_path_safe = Path(path) / "model.safetensors"
            if state_path_bin.exists():
                from torch import load as torch_load
                full_state = torch_load(state_path_bin, map_location=dev)
            elif state_path_safe.exists():
                from safetensors.torch import load_file  # type: ignore
                full_state = load_file(state_path_safe, device=str(dev))
            else:
                raise FileNotFoundError(f"No weight file (pytorch_model.bin / model.safetensors) found in {path}")

            base_state = {k: v for k, v in full_state.items() if not k.startswith("crf.")}

            # ------------------------------------------------------------------
            # Build model configuration first – allows us to load the model
            # without also downloading the original checkpoint weights.
            # ------------------------------------------------------------------
            config = AutoConfig.from_pretrained(
                params["model_checkpoint"],
                num_labels=len(label_names),
                id2label=inst.id2label,
                label2id=inst.label2id,
            )

            # ------------------------------------------------------------------
            # Initialise backbone WITHOUT passing the state_dict to avoid the
            # "state_dict cannot be passed together with a model name" error
            # introduced in recent 🤗 Transformers releases. We then load the
            # weights manually with strict=False so that the classification head
            # size mismatch is tolerated.
            # ------------------------------------------------------------------
            from transformers import AutoModelForTokenClassification

            inst.model = AutoModelForTokenClassification.from_pretrained(
                params["model_checkpoint"],
                config=config,
                ignore_mismatched_sizes=True,
            ).to(dev)

            # Manually load our fine-tuned weights (excluding CRF)
            missing, unexpected = inst.model.load_state_dict(base_state, strict=False)
            if unexpected:
                print("⚠️ Unexpected keys when loading backbone state:", unexpected)
            if missing:
                print("⚠️ Missing keys when loading backbone state:", missing)

            # ------------------------------
            # Restore CRF weights
            # ------------------------------
            from torchcrf import CRF  # type: ignore
            inst.model.crf = CRF(num_tags=len(label_names), batch_first=True).to(dev)

            crf_state = {k.replace("crf.", ""): v for k, v in full_state.items() if k.startswith("crf.")}
            missing, unexpected = inst.model.crf.load_state_dict(crf_state, strict=False)
            if missing or unexpected:
                print("⚠️ CRF state mismatch while loading model (safe to ignore if keys are zero):", missing, unexpected)

            # ------------------------------
            # Prediction aggregator
            # ------------------------------
            inst.prediction_aggregator = PredictionAggregator(inst)

            return inst

    # ------------------------------------------------------------------
    # Override window forward pass to utilise CRF decoding
    # ------------------------------------------------------------------
    def _forward_windows(self, encodings):  # type: ignore[override]
        all_preds, all_probs, all_offsets = [], [], []
        for i in range(encodings["input_ids"].size(0)):
            inputs = {k: v[i].unsqueeze(0).to(device) for k, v in encodings.items() if k in ["input_ids", "attention_mask"]}
            with torch.no_grad():
                # Get raw emissions (logits) without softmax
                emissions = self.model(**inputs, return_dict=True).logits  # Raw emissions (1, L, C)

            # Decode with CRF if present, using raw emissions
            if hasattr(self.model, "crf") and self.model.crf is not None:
                mask = inputs["attention_mask"].bool()
                decoded = self.model.crf.decode(emissions, mask=mask)[0]
                seq_len = emissions.shape[1]
                pred_arr = np.zeros(seq_len, dtype=int)
                pred_arr[: len(decoded)] = decoded
            else:
                pred_arr = emissions.argmax(dim=-1).squeeze(0).cpu().numpy()

            # Compute probabilities for span merging
            probs = torch.softmax(emissions, dim=-1)[0].cpu().numpy()
            all_preds.append(pred_arr)
            all_probs.append(probs)
            all_offsets.append(encodings["offset_mapping"][i].cpu().numpy())
        return all_preds, all_probs, all_offsets

    def _validate_entity(self, span):  # type: ignore[override]
        word = span["word"].strip()

        latin_only = all('LATIN' in unicodedata.name(ch, '') for ch in word if ch.isalpha())
        if latin_only and not any(c.isupper() for c in word):
            return False

        # reject very short tokens
        if len(word) < 2:
            return False

        # reject punctuation-only strings
        if all(not ch.isalnum() for ch in word):
            return False

        w_lower = word.lower()
        function_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'must', 'this',
            'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my',
            'your', 'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'him',
            'her', 'us', 'them'
        }
        if w_lower in function_words:
            return False

        # basic probability floor (already scaled)
        max_p = max(span["prob_antagonist"], span["prob_protagonist"], span["prob_innocent"])
        if max_p < 0.3:
            return False

        return True

# -----------------------------------------------------------------------------
# Span merger with per-label scaling and stricter validation (shared by training
# and by the static `load()` method so it must live at module scope)
# -----------------------------------------------------------------------------


class ScaledSpanMerger(_SpanMerger):
    """_SpanMerger that (1) multiplies class-specific confidence scores before
    thresholding and (2) applies stricter span validation rules to curb false
    positives.
    """

    DEFAULT_SCALES = {
        "Antagonist": 1.0,
        "Protagonist": 1.2,
        "Innocent": 1.5,
    }

    def __init__(self, *args, label_scales: dict[str, float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_scales = label_scales or self.DEFAULT_SCALES

    # ---------------------------
    # Span-probability aggregation
    # ---------------------------
    def _flush(self, cur, spans, raw_text):  # type: ignore[override]
        if cur is None:
            return
        probs = np.mean(cur["probs"], axis=0)

        # Get raw probabilities (before scaling)
        raw_ant = float(probs[self.ant_ids].sum())
        raw_prot = float(probs[self.prot_ids].sum())
        raw_inn = float(probs[self.inn_ids].sum())
        raw_o = float(probs[self.label2id["O"]])
        raw_unk = float(probs[self.unk_ids].sum()) if self.unk_ids else 0.0

        # Apply scaling for thresholding decision
        scaled_ant = raw_ant * self.label_scales.get("Antagonist", 1.0)
        scaled_prot = raw_prot * self.label_scales.get("Protagonist", 1.0)
        scaled_inn = raw_inn * self.label_scales.get("Innocent", 1.0)
        scaled_max = max(scaled_ant, scaled_prot, scaled_inn, raw_unk)

        # Check if any scaled score beats O and threshold
        if scaled_max <= raw_o or scaled_max <= self.threshold:
            return

        # For final probabilities: normalize the RAW scores (not scaled)
        # then apply light smoothing to prevent extreme values
        tot = raw_ant + raw_prot + raw_inn
        if tot == 0:
            return

        ant = raw_ant / tot
        prot = raw_prot / tot  
        inn = raw_inn / tot
        unk = (raw_unk / tot) if self.unk_ids else 0.0

        # Apply smoothing to prevent 1.00/0.00 extremes (focal loss artifact)
        # Use temperature scaling instead of hard clamping to preserve relative differences
        temperature = 2.0  # Higher temperature = softer probabilities
        
        # Convert to logits, apply temperature, then back to probabilities
        import math
        ant_logit = math.log(max(ant, 1e-8))
        prot_logit = math.log(max(prot, 1e-8))
        inn_logit = math.log(max(inn, 1e-8))
        
        # Apply temperature scaling
        ant_logit /= temperature
        prot_logit /= temperature
        inn_logit /= temperature
        
        # Convert back to probabilities
        max_logit = max(ant_logit, prot_logit, inn_logit)
        ant_exp = math.exp(ant_logit - max_logit)
        prot_exp = math.exp(prot_logit - max_logit)
        inn_exp = math.exp(inn_logit - max_logit)
        
        total_exp = ant_exp + prot_exp + inn_exp
        ant = ant_exp / total_exp
        prot = prot_exp / total_exp
        inn = inn_exp / total_exp

        # Dirichlet smoothing to ensure none of the probabilities hit 0/1 exactly
        alpha = 0.05
        ant = (ant + alpha) / (1 + 3 * alpha)
        prot = (prot + alpha) / (1 + 3 * alpha)
        inn = (inn + alpha) / (1 + 3 * alpha)

        span = {
            "start": cur["start"],
            "end": cur["end"],
            "word": raw_text[cur["start"]:cur["end"]],
            "prob_antagonist": ant,
            "prob_protagonist": prot,
            "prob_innocent": inn,
            "prob_unknown": unk,
        }

        span = snap_boundaries(span, raw_text)
        if self._validate_entity(span):
            spans.append(span)

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
                # Merge if same type and close, OR if very close (likely same name)
                if (current_best == next_best and 0 <= gap <= 5) or (0 <= gap <= 2):
                    # Merge spans
                    current["end"] = next_span["end"]
                    current["word"] = raw_text[current["start"]:current["end"]]
                    
                    # If same type, average probabilities; if different types, use the higher confidence one
                    if current_best == next_best:
                        # Average the probabilities
                        current["prob_antagonist"] = (current["prob_antagonist"] + next_span["prob_antagonist"]) / 2
                        current["prob_protagonist"] = (current["prob_protagonist"] + next_span["prob_protagonist"]) / 2
                        current["prob_innocent"] = (current["prob_innocent"] + next_span["prob_innocent"]) / 2
                        current["prob_unknown"] = (current["prob_unknown"] + next_span["prob_unknown"]) / 2
                    else:
                        # Use the span with higher confidence
                        current_max_prob = max(current["prob_antagonist"], current["prob_protagonist"], current["prob_innocent"])
                        next_max_prob = max(next_span["prob_antagonist"], next_span["prob_protagonist"], next_span["prob_innocent"])
                        
                        if next_max_prob > current_max_prob:
                            # Use next span's probabilities
                            current["prob_antagonist"] = next_span["prob_antagonist"]
                            current["prob_protagonist"] = next_span["prob_protagonist"]
                            current["prob_innocent"] = next_span["prob_innocent"]
                            current["prob_unknown"] = next_span["prob_unknown"]
                    
                    j += 1  # Skip the merged span
                else:
                    break  # No more spans to merge
            
            merged.append(current)
            i = j if j > i + 1 else i + 1  # Move to next unprocessed span
        
        return merged

    def merge(self, raw_text: str, preds: np.ndarray, probs: np.ndarray, offsets: np.ndarray):
        """Override merge to add post-processing for nearby span merging."""
        # Call parent merge method first
        spans = super().merge(raw_text, preds, probs, offsets)
        
        # Apply post-processing to merge nearby spans of the same type
        spans = self._merge_nearby_spans(spans, raw_text)
        
        return spans

    # ---------------------------
    # Stricter validation rules
    # ---------------------------
    def _validate_entity(self, span):  # type: ignore[override]
        word = span["word"].strip()

        latin_only = all('LATIN' in unicodedata.name(ch, '') for ch in word if ch.isalpha())
        if latin_only and not any(c.isupper() for c in word):
            return False

        # reject very short tokens
        if len(word) < 2:
            return False

        # reject punctuation-only strings
        if all(not ch.isalnum() for ch in word):
            return False

        w_lower = word.lower()
        function_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'must', 'this',
            'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my',
            'your', 'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'him',
            'her', 'us', 'them'
        }
        if w_lower in function_words:
            return False

        # basic probability floor (already scaled)
        max_p = max(span["prob_antagonist"], span["prob_protagonist"], span["prob_innocent"])
        if max_p < 0.3:
            return False

        return True