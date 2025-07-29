import os
import pandas as pd
import nltk
import streamlit as st
from nltk.tokenize import sent_tokenize
import re

# Lazy NLTK download - only when needed
def ensure_nltk_data():
    """Download NLTK data if not already present"""
    try:
        # Try to use punkt tokenizer - if it fails, download it
        sent_tokenize("Test sentence.")
    except LookupError:
        print("📥 Downloading NLTK punkt tokenizer...")
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            print(f"⚠️ NLTK download failed: {e}")
            # Fallback to basic splitting if NLTK fails
            pass

def char_window_context(text, start_offset, end_offset, window=150):
    try:
        start_offset = int(start_offset)
        end_offset = int(end_offset)
    except (ValueError, TypeError):
        return text[:2 * window].strip()

    left = max(0, start_offset - window)
    right = min(len(text),end_offset + window)
    return text[left:right].strip()

def extract_entity_sentence(article_text, start_offset, end_offset):
    try:
        start_offset = int(start_offset)
        end_offset = int(end_offset)
 
       
        sentences = re.split(r'(?<=[.!?])\s+', article_text)
 
        
        current_pos = 0
        for sentence in sentences:
            sentence_len = len(sentence)
            if current_pos <= start_offset < current_pos + sentence_len:
                return sentence.strip()
            current_pos += sentence_len + 1  # +1 for the split whitespace
        return ""
    except Exception as e:
        return f"Error: {e}"


def convert_prediction_txt_to_csv(article_id, article, prediction_file, article_text, output_csv):
    """
    Converts a Stage 1 prediction .txt into a CSV with full article text.
    Adds the provided article_id to each row.
    """
    records = []
    with open(prediction_file, encoding="utf-8") as f:
        for line in f:
            ##st.write(line)
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue  # Skip incomplete lines
            article_id, entity, start, end, p_main_role = parts[:5]


            context = char_window_context(article, start, end)

            sentence = extract_entity_sentence(article, start, end)

            records.append({
                "article_id": article_id,
                "text": article,
                "entity_mention": entity,
                "start_offset": start,
                "end_offset": end,
                "p_main_role": p_main_role,
                "context": context,
                "sentence": sentence
            })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8")

