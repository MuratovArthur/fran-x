#!/usr/bin/env python3

"""Standalone script for NER prediction with DeBERTa model.

This script provides two main functionalities:

1. NEW: Process raw text input and generate predictions saved to txt files
   - Use predict_from_text() or predict_text_to_file() functions
   - Input: raw text string
   - Output: txt file with tab-separated predictions (entity, start, end, role)

2. LEGACY: Process txt files from dataset directories (original functionality)
   - Runs inference_deberta and reproduces exact-match results from notebook
   - Use main_legacy() function or process_language() directly

Usage examples:
   # For raw text prediction:
   python predict.py  # runs main() with sample text
   
   # Or import and use:
   from predict import predict_text_to_file
   predictions, count = predict_text_to_file("Your text here", "output.txt")
"""

import sys
import json
import string
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import inference_deberta only when needed (for legacy functions)
try:
    from scripts.inference import inference_deberta
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    inference_deberta = None


def predict_from_text(bert_model, text, output_filename="predictions.txt", output_dir="output"):
    """Process raw text and save predictions to a txt file."""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get predictions from the model
    spans = bert_model.predict(text, return_format='spans')
    pred_spans = []
    
    for sp in spans:
        s, e = sp['start'], sp['end']
        seg = text[s:e]
        s += len(seg) - len(seg.lstrip())
        e -= len(seg) - len(seg.rstrip())
        role_probs = [(sp['prob_antagonist'], 'Antagonist'),
                      (sp['prob_protagonist'], 'Protagonist'),
                      (sp['prob_innocent'], 'Innocent'),
                      (sp['prob_unknown'], 'Unknown')]
        _, role = max(role_probs)
        pred_spans.append((s, e, role))

    # Format predictions for output
    output_lines = []
    non_unknown = 0
    
    for s, e, role in pred_spans:
        entity_text = text[s:e].replace('\n', ' ').replace('\r', ' ').strip()
        if role != 'Unknown':
            non_unknown += 1
        # Format: entity_text, start, end, role
        output_lines.append(f"{entity_text}\t{s}\t{e}\t{role}")

    # Save predictions to txt file
    output_file_path = output_path / output_filename
    output_file_path.write_text('\n'.join(output_lines), encoding='utf-8')
    
    print(f"Saved {len(output_lines)} predictions to {output_file_path}")
    print(f"Non-'Unknown' entries: {non_unknown}")
    
    return output_lines, non_unknown


def main():
    """Main function with example usage of predict_from_text."""
    try:
        import torch
        from src.deberta import DebertaV3NerClassifier
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure you have the 'src' directory with 'deberta.py' containing DebertaV3NerClassifier")
        return

    # Load model
    model_path = 'artur-muratov/franx-ner'
    bert_model = DebertaV3NerClassifier.load(model_path)

    # Add +1 bias to non-O classes (same as inference_deberta)
    with torch.no_grad():
        current_bias = bert_model.model.classifier.bias
        o_index = bert_model.label2id.get('O', 0)
        for i in range(len(current_bias)):
            if i != o_index:
                current_bias[i] += 1.0

    bert_model.model = bert_model.model.to('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(bert_model, 'merger'):
        bert_model.merger.threshold = 0.5

    # Example usage with raw text
    sample_text = """
    John Smith was involved in the incident. The victim, Mary Johnson, reported the case to the police. 
    The suspect fled the scene and is still at large. Detective Brown is investigating the case.
    """
    
    print("Processing sample text...")
    print("=" * 60)
    print(f"Input text: {sample_text.strip()}")
    print("-" * 60)
    
    # Predict from raw text and save to file
    predictions, non_unknown_count = predict_from_text(
        bert_model, 
        sample_text, 
        output_filename="sample_predictions.txt",
        output_dir="predictions_output"
    )
    
    print(f"\nPredictions found:")
    for pred in predictions:
        entity, start, end, role = pred.split('\t')
        print(f"  [{start}:{end}] {role} = '{entity}'")
    
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Non-'Unknown' predictions: {non_unknown_count}")


def predict_text_to_file(text, output_filename="predictions.txt", output_dir="output", model_path=None):
    """
    Convenience function to predict from text and save to file.
    
    Args:
        text (str): Input text to analyze
        output_filename (str): Name of output file
        output_dir (str): Directory to save output file
        model_path (str): Path to model (optional, uses default if None)
    
    Returns:
        tuple: (predictions_list, non_unknown_count)
    """
    try:
        import torch
        from src.deberta import DebertaV3NerClassifier
    except ImportError as e:
        raise ImportError(f"Could not import required modules: {e}. Please ensure you have the 'src' directory with 'deberta.py' containing DebertaV3NerClassifier")
    
    # Load model
    if model_path is None:
        model_path = 'artur-muratov/franx-ner'
    
    bert_model = DebertaV3NerClassifier.load(model_path)
    
    # Add +1 bias to non-O classes (same as inference_deberta)
    with torch.no_grad():
        current_bias = bert_model.model.classifier.bias
        o_index = bert_model.label2id.get('O', 0)
        for i in range(len(current_bias)):
            if i != o_index:
                current_bias[i] += 1.0
    
    bert_model.model = bert_model.model.to('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(bert_model, 'merger'):
        bert_model.merger.threshold = 0.5
    
    # Make predictions
    return predict_from_text(bert_model, text, output_filename, output_dir)


if __name__ == '__main__':
    main() 