# scripts/train.py

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from collections import Counter

from src.bert import BertNerClassifier
from src.schema import BERTDataset
from src.deberta import DebertaV3NerClassifier


def train_bert(
        train_raw,
        val_raw,
        label_list,
        experiment_name,
        model_name="bert-base-cased",
        n_epochs=5,
        output_model_path="ner_trained/bert_checkpoint"
):
    """Train a BERT model for NER.
    
    Args:
        train_raw: List of training documents with tokens and labels
        val_raw: List of validation documents with tokens and labels
        label_list: List of possible labels
        experiment_name: Name of the experiment/run
        model_name: Name or path of the base BERT model to use
        n_epochs: Number of training epochs
        output_model_path: Directory to save the trained model
    """
    # Use the improved parameters from BertNerClassifier
    params = dict(num_train_epochs=n_epochs)

    bert_model = BertNerClassifier(
        label_names=label_list, model_checkpoint=model_name
    )

    # Convert JSON documents to the expected format
    train_tokens = [
        [t["text"] if isinstance(t["text"], str) else "" for t in train_doc]
        for train_doc in train_raw
    ]
    train_ner_tags = [
        [bert_model.label2id[t["bio_label"]] for t in train_doc]
        for train_doc in train_raw
    ]
    raw_dataset_train: BERTDataset = {
        "tokens": train_tokens,
        "ner_tags": train_ner_tags,
    }

    val_tokens = [
        [t["text"] if isinstance(t["text"], str) else "" for t in val_doc]
        for val_doc in val_raw
    ]
    val_ner_tags = [
        [bert_model.label2id[t["bio_label"]] for t in val_doc] for val_doc in val_raw
    ]
    raw_dataset_val: BERTDataset = {"tokens": val_tokens, "ner_tags": val_ner_tags}

    # Print dataset statistics
    print(f"📊 Training dataset: {len(train_raw)} documents, {sum(len(doc) for doc in train_tokens)} total tokens")
    print(f"📊 Validation dataset: {len(val_raw)} documents, {sum(len(doc) for doc in val_tokens)} total tokens")

    bert_model.train(
        model_name=f"{output_model_path}/{experiment_name}",
        train_data=raw_dataset_train,
        val_data=raw_dataset_val,
        **params,
    )

    bert_model.save(f"{output_model_path}/{experiment_name}")


def train_deberta(
        train_raw,
        val_raw,
        label_list,
        experiment_name,
        model_name="microsoft/deberta-v3-base",
        output_model_path="ner_trained/deberta_v3_checkpoint",
        **train_kwargs,
):
    """Train a DeBERTa v3 model for NER."""
    n_epochs = train_kwargs.pop("n_epochs", 5)
    params = dict(num_train_epochs=n_epochs)
    # propagate focal flag etc.
    params.update(train_kwargs)

    deberta_model = DebertaV3NerClassifier(
        label_names=label_list, model_checkpoint=model_name
    )

    train_tokens = [
        [t["text"] if isinstance(t["text"], str) else "" for t in train_doc]
        for train_doc in train_raw
    ]
    train_ner_tags = [
        [deberta_model.label2id[t["bio_label"]] for t in train_doc]
        for train_doc in train_raw
    ]
    raw_dataset_train: BERTDataset = {
        "tokens": train_tokens,
        "ner_tags": train_ner_tags,
    }

    val_tokens = [
        [t["text"] if isinstance(t["text"], str) else "" for t in val_doc]
        for val_doc in val_raw
    ]
    val_ner_tags = [
        [deberta_model.label2id[t["bio_label"]] for t in val_doc]
        for val_doc in val_raw
    ]
    raw_dataset_val: BERTDataset = {"tokens": val_tokens, "ner_tags": val_ner_tags}

    print(f"📊 Training dataset for DeBERTa: {len(train_raw)} documents, {sum(len(doc) for doc in train_tokens)} total tokens")
    print(f"📊 Validation dataset for DeBERTa: {len(val_raw)} documents, {sum(len(doc) for doc in val_tokens)} total tokens")

    model_dir = Path(f"{output_model_path}/{experiment_name}")
    deberta_model.train(
        model_name=str(model_dir),
        train_data=raw_dataset_train,
        val_data=raw_dataset_val,
        **params,
    )

    deberta_model.save(str(model_dir))

    # ------------------------------------------------------------------
    # Save training metadata for reproducibility
    # ------------------------------------------------------------------
    metadata = {
        "model_checkpoint": model_name,
        "experiment_name": experiment_name,
        "num_epochs": n_epochs,
        "train_docs": len(train_raw),
        "val_docs": len(val_raw),
        "train_tokens": sum(len(doc) for doc in train_raw),
        "val_tokens": sum(len(doc) for doc in val_raw),
        "label_list": label_list,
        "use_focal": params.get("use_focal", False),
        "focal_gamma": params.get("focal_gamma", 2.0) if params.get("use_focal", False) else None,
        "training_args": deberta_model.trainer.args.to_dict(),
    }

    with open(model_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save full training logs (trainer log history)
    log_history = deberta_model.trainer.state.log_history
    with open(model_dir / "training_log_history.json", "w", encoding="utf-8") as f:
        json.dump(log_history, f, default=str, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT model for NER")
    parser.add_argument(
        "--type",
        "-t",
        help="Model type",
        default="bert",
        const="bert",
        nargs="?",
        choices=["bert", "deberta"],
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Path to model or model name",
        default="bert-base-cased",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        help="Path to dataset directory containing bio/ folder with JSON files",
        default="dataset/train/EN",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        help="Number of epochs to run training for",
        default=5,
    )
    parser.add_argument(
        "--experiment",
        "-x",
        type=str,
        help="Name of the training experiment",
        default="first_run",
    )
    parser.add_argument(
        "--focal",
        action="store_true",
        help="Use focal loss instead of cross entropy",
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save training metadata (dataset stats, hyperparameters) to JSON",
    )

    args = parser.parse_args()

    # Load JSON documents and extract language codes
    bio_dir = Path(args.data) / "bio"
    file_paths = sorted(bio_dir.glob("*.json"))
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in file_paths]
    # Determine language for each document by checking substrings
    supported_langs = ["BG", "EN", "HI", "PT", "RU"]
    langs = []
    for p in file_paths:
        fname = p.name
        # Assign based on substring match
        for lang in supported_langs:
            if lang in fname:
                langs.append(lang)
                break
        else:
            langs.append("OTHER")
            print(f"⚠️ Could not determine language for file: {fname}")
    print(f"Loaded {len(docs)} JSON documents from {bio_dir}")
    
    # Determine if stratified split is feasible; skip stratify for languages with <2 docs
    lang_counts = Counter(langs)
    small_langs = [lang for lang, count in lang_counts.items() if count < 2]
    if small_langs:
        print(f"⚠️ Languages with fewer than 2 docs {small_langs}, using random split without stratification.")
        docs_train, docs_val, langs_train, langs_val = train_test_split(
            docs,
            langs,
            test_size=0.2,
            random_state=42
        )
    else:
        docs_train, docs_val, langs_train, langs_val = train_test_split(
            docs,
            langs,
            test_size=0.2,
            random_state=42,
            stratify=langs
        )
    # Assign train and validation datasets
    train_raw, val_raw = docs_train, docs_val
    # Print per-language split counts
    print("📊 Document split by language:")
    print("  Training set:")
    for lang, cnt in Counter(langs_train).items():
        print(f"    {lang}: {cnt}")
    print("  Validation set:")
    for lang, cnt in Counter(langs_val).items():
        print(f"    {lang}: {cnt}")

    # Define label list
    label_list = [
        "O",
        "B-Antagonist", "I-Antagonist",
        "B-Protagonist", "I-Protagonist",
        "B-Innocent", "I-Innocent"
    ]

    # Train the model
    train_kwargs = {
        "n_epochs": args.epochs,
    }
    if args.focal:
        train_kwargs["use_focal"] = True

    if args.type == "bert":
        train_bert(
            train_raw,
            val_raw,
            label_list,
            experiment_name=args.experiment,
            model_name=args.model,
            **train_kwargs,
        )
    elif args.type == "deberta":
        train_deberta(
            train_raw,
            val_raw,
            label_list,
            experiment_name=args.experiment,
            model_name=args.model,
            **train_kwargs,
        )