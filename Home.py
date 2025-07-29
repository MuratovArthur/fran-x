import streamlit as st
import pandas as pd
import altair as alt
import requests
import datetime
import re
import sys
import os
from pathlib import Path
import ast
import secrets
from dotenv import load_dotenv
from sidebar import render_sidebar, ROLE_COLORS
from render_text import reformat_text_html_with_tooltips, predict_entity_framing, format_sentence_with_spans
from mode_tc_utils.preprocessing import convert_prediction_txt_to_csv
from mode_tc_utils.tc_inference import run_role_inference
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# ============================================================================
# MODEL LOADING FUNCTIONS (optimized for performance)
# ============================================================================

# Configuration for optimized inference
def get_inference_url():
    """Get inference service URL from secrets or environment"""
    try:
        return st.secrets.get('INFERENCE_URL', 'http://localhost:8000')
    except:
        return os.getenv('INFERENCE_URL', 'http://localhost:8000')

API_BASE_URL = get_inference_url()

def check_model_service():
    """Check if the model service is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Service returned {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, str(e)

@st.cache_data(ttl=60)  # Cache service check for 1 minute  
def get_service_status():
    """Get model service status with caching"""
    return check_model_service()

def _run_ner_inference(text):
    """Run NER model inference"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ner",
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ NER model failed to load or process text. Please try again.")
        return None

def _run_classification_inference(entity_mention, p_main_role, context, threshold=0.01, margin=0.05):
    """Run classification model inference"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/classify",
            json={
                "entity_mention": entity_mention,
                "p_main_role": p_main_role,
                "context": context,
                "threshold": threshold,
                "margin": margin
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Classification model failed to load or process text. Please try again.")
        return None

# Optimized model classes for efficient inference
class OptimizedNERModel:
    """Optimized NER model for efficient inference"""
    def predict(self, text, return_format='spans'):
        result = _run_ner_inference(text)
        if result is None:
            return []
        return result

class OptimizedClassifier:
    """Optimized classifier for efficient inference"""
    def __call__(self, input_text):
        # Parse the input to extract components
        lines = input_text.strip().split('\n')
        entity_mention = lines[0].replace('Entity: ', '') if len(lines) > 0 else ""
        p_main_role = lines[1].replace('Main Role: ', '') if len(lines) > 1 else ""
        context = lines[2].replace('Context: ', '') if len(lines) > 2 else ""
        
        result = _run_classification_inference(entity_mention, p_main_role, context)
        if result is None:
            return [[]]
        
        # Convert to expected pipeline format
        scores = result.get('predicted_fine_with_scores', {})
        pipeline_format = [[{"label": label, "score": score} for label, score in scores.items()]]
        return pipeline_format

def get_stage2_model():
    """Load optimized classification model"""
    return OptimizedClassifier()

def get_ner_model():
    """Load optimized NER model"""
    return OptimizedNERModel()

# Utility functions for prediction and stage2

def predict_with_cached_model(article_id, bert_model, text, output_dir="output"):
    spans = bert_model.predict(text, return_format='spans')
    pred_spans = []
    for sp in spans:
        s, e = sp['start'], sp['end']
        seg = text[s:e]
        s += len(seg) - len(seg.lstrip())
        e -= len(seg) - len(seg.rstrip())
        role_probs = [
            (sp['prob_antagonist'], 'Antagonist'),
            (sp['prob_protagonist'], 'Protagonist'),
            (sp['prob_innocent'], 'Innocent'),
            (sp['prob_unknown'], 'Unknown')
        ]
        _, role = max(role_probs)
        pred_spans.append((s, e, role))

    output_lines = []
    non_unknown = 0
    for s, e, role in pred_spans:
        entity_text = text[s:e].strip().replace('\n', ' ')
        if role != 'Unknown':
            non_unknown += 1
        output_lines.append(f"{article_id}\t{entity_text}\t{s}\t{e}\t{role}")

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    file_path = out_dir / f"{article_id}_predictions.txt"
    file_path.write_text("\n".join(output_lines), encoding='utf-8')
    return output_lines, non_unknown


def run_stage2_with_cached_model(article_id, clf_pipeline, df, threshold=0.01, margin=0.05):
    def pipeline_with_confidence(example):
        input_text = (
            f"Entity: {example['entity_mention']}\n"
            f"Main Role: {example['p_main_role']}\n"
            f"Context: {example['context']}"
        )
        scores = clf_pipeline(input_text)[0]
        return {s['label']: round(s['score'], 4) for s in scores if s['score'] > threshold}

    def select_roles(scores):
        if not scores:
            return []
        top_score = max(scores.values())
        return [role for role, score in scores.items() if score >= top_score - margin]

    def filter_scores(row):
        scores = row['predicted_fine_with_scores']
        roles = row['predicted_fine_margin']
        return {r: scores[r] for r in roles}

    df['predicted_fine_with_scores'] = df.apply(pipeline_with_confidence, axis=1)
    df['predicted_fine_margin'] = df['predicted_fine_with_scores'].apply(select_roles)
    df['p_fine_roles_w_conf'] = df.apply(filter_scores, axis=1)
    df['article_id'] = article_id
    return df

# ----------------------------------------------------------------------------
# Streamlit App Layout and Logic
# ----------------------------------------------------------------------------
st.set_page_config(page_title="FRaN-X", layout="wide")
st.title("FRaN-X: Entity Framing & Narrative Analysis")

# Session management
if 'session_id' not in st.session_state:
    st.session_state.session_id = secrets.token_hex(4)
user_folder = st.session_state.session_id
st.info(f"Your session ID: `{user_folder}`. Save this to revisit your files.")

# Article input
filename_input = st.text_input("Filename (without extension)")
mode = st.radio("Input mode", ["Paste Text", "URL"])
article = ""
if mode == "Paste Text":
    article = st.text_area("Article", height=300)
else:
    url = st.text_input("Article URL")
    if url:
        try:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.content, 'html.parser')
            article = '\n'.join(p.get_text() for p in soup.find_all('p'))
        except:
            st.error("Failed to fetch article. Please paste text instead.")

if article:
    st.caption(f"Article length: {len(article)} characters")

# Prediction trigger
if st.button("Run Entity Predictions"):
    if not filename_input:
        st.warning("Enter a filename before running predictions.")
    elif not article.strip():
        st.warning("Provide article text or valid URL.")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        article_id = f"{filename_input}_{timestamp}"

        # Load models (lightweight - optimized for performance)
        with st.spinner("Loading classification model..."):
            clf_pipeline = get_stage2_model()

        with st.spinner("Loading NER model..."):
            bert_model = get_ner_model()

        # Stage 1 predictions
        with st.spinner("Analyzing entities..."):
            preds, non_unknown_count = predict_with_cached_model(article_id, bert_model, article, output_dir="article_predictions")
        st.success(f"Found {len(preds)} entities ({non_unknown_count} with specific roles)")

        # Prepare and run Stage 2
        convert_prediction_txt_to_csv(
            article_id=article_id,
            article=article,
            prediction_file=f"article_predictions/{article_id}_predictions.txt",
            article_text=article,
            output_csv="article_predictions/tc_input.csv"
        )
        input_df = pd.read_csv("article_predictions/tc_input.csv")
        stage2_df = run_stage2_with_cached_model(article_id, clf_pipeline, input_df)
        stage2_df.to_csv("article_predictions/tc_output.csv", index=False)

        # Display results
        with st.expander("Detected Entities", expanded=True):
            spans = bert_model.predict(article, return_format='spans')
            for line in preds:
                _, entity, s, e, role = line.split('\t')
                s, e = int(s), int(e)
                conf = next((sp[f"prob_{role.lower()}"] for sp in spans if sp['start']==s and sp['end']==e), None)
                icon = {'Protagonist':'🟢','Antagonist':'🔴','Innocent':'🔵','Unknown':'⚪'}.get(role)
                st.markdown(f"{icon} **{entity}** - {role} (confidence: {conf:.3f}) [{s}-{e}]")

        with st.expander("Fine-Grained Role Predictions", expanded=True):
            for _, row in stage2_df.iterrows():
                roles = row['predicted_fine_margin']
                scores = row['predicted_fine_with_scores']
                formatted = ", ".join(f"{r}: {scores.get(r,0):.3f}" for r in roles)
                st.markdown(f"**{row['entity_mention']}** ({row['p_main_role']}): _{formatted}_")

st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team*")