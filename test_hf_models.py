#!/usr/bin/env python3
"""Test script to verify Hugging Face model loading."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_ner_model():
    """Test loading the NER model from Hugging Face."""
    print("Testing NER model loading...")
    try:
        from src.deberta import DebertaV3NerClassifier
        
        model_path = 'artur-muratov/franx-ner'
        print(f"Loading NER model from: {model_path}")
        
        bert_model = DebertaV3NerClassifier.load(model_path)
        print(f"✅ NER model loaded successfully!")
        print(f"   Model checkpoint: {bert_model.model_checkpoint}")
        print(f"   Label names: {bert_model.label_names}")
        print(f"   Max length: {bert_model.max_length}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load NER model: {e}")
        return False

def test_cls_model():
    """Test loading the classification model from Hugging Face."""
    print("\nTesting classification model loading...")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        
        model_path = "artur-muratov/franx-cls"
        print(f"Loading classification model from: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        clf_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        
        print(f"✅ Classification model loaded successfully!")
        print(f"   Model config: {model.config.architectures}")
        print(f"   Num labels: {model.config.num_labels}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load classification model: {e}")
        return False

def test_prediction():
    """Test a simple prediction with both models."""
    print("\nTesting prediction pipeline...")
    try:
        # Test text
        test_text = "John Smith was involved in the incident. The police are investigating."
        
        # Load NER model
        from src.deberta import DebertaV3NerClassifier
        bert_model = DebertaV3NerClassifier.load('artur-muratov/franx-ner')
        
        # Run prediction
        spans = bert_model.predict(test_text, return_format='spans')
        print(f"✅ Prediction successful!")
        print(f"   Found {len(spans)} spans")
        for span in spans[:3]:  # Show first 3
            print(f"   - {span['word']} ({span['start']}-{span['end']})")
        
        return True
    except Exception as e:
        print(f"❌ Failed to run prediction: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Hugging Face model integration...")
    print("=" * 50)
    
    results = []
    results.append(test_ner_model())
    results.append(test_cls_model())
    results.append(test_prediction())
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Hugging Face integration is working.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.") 