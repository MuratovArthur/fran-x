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
# MODEL LOADING FUNCTIONS (cached)
# ============================================================================
@st.cache_resource
def get_stage2_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from huggingface_hub import login, HfFolder
    import os
    import streamlit as st

    # Authenticate with HF token
    hf_token = st.secrets.get('HF_TOKEN') or os.getenv('HF_TOKEN')
    if hf_token:
        login(token=hf_token, write_permission=False)

    model_id = "artur-muratov/franx-cls"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

@st.cache_resource
def get_ner_model():
    import torch
    from src.deberta import DebertaV3NerClassifier
    from huggingface_hub import login, HfFolder
    import os
    import streamlit as st

    # Authenticate with HF token
    hf_token = st.secrets.get('HF_TOKEN') or os.getenv('HF_TOKEN')
    if hf_token:
        login(token=hf_token, write_permission=False)

    model_id = 'artur-muratov/franx-ner'
    bert_model = DebertaV3NerClassifier.load(model_id)

    # bias adjustment
    with torch.no_grad():
        bias = bert_model.model.classifier.bias
        o_index = bert_model.label2id.get('O', 0)
        for i in range(len(bias)):
            if i != o_index:
                bias[i] += 1.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_model.model.to(device)
    if hasattr(bert_model, 'merger'):
        bert_model.merger.threshold = 0.5
    return bert_model

# Utility functions

def predict_with_cached_model(article_id, bert_model, text, output_dir="output"):
    from pathlib import Path
    spans = bert_model.predict(text, return_format='spans')
    pred_spans = []

    for sp in spans:
        s, e = sp['start'], sp['end']
        # trim whitespace
        seg = text[s:e]
        s += len(seg) - len(seg.lstrip())
        e -= len(seg) - len(seg.rstrip())
        role_probs = [(sp['prob_antagonist'], 'Antagonist'),
                      (sp['prob_protagonist'], 'Protagonist'),
                      (sp['prob_innocent'], 'Innocent'),
                      (sp['prob_unknown'], 'Unknown')]
        _, role = max(role_probs)
        pred_spans.append((s, e, role))

    output_lines = []
    non_unknown = 0
    for s, e, role in pred_spans:
        text_seg = text[s:e].strip().replace('\n', ' ')
        if role != 'Unknown':
            non_unknown += 1
        output_lines.append(f"{article_id}\t{text_seg}\t{s}\t{e}\t{role}")

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
        top = max(scores.values())
        return [r for r, sc in scores.items() if sc >= top - margin]

    def filter_scores(row):
        scores = row['predicted_fine_with_scores']
        roles = row['predicted_fine_margin']
        return {r: scores[r] for r in roles}

    df['predicted_fine_with_scores'] = df.apply(pipeline_with_confidence, axis=1)
    df['predicted_fine_margin'] = df['predicted_fine_with_scores'].apply(select_roles)
    df['p_fine_roles_w_conf'] = df.apply(filter_scores, axis=1)
    df['article_id'] = article_id
    return df

# Streamlit App
st.set_page_config(page_title="FRaN-X", layout="wide")
st.title("FRaN-X: Entity Framing & Narrative Analysis")

# Session and file management
if 'session_id' not in st.session_state:
    st.session_state.session_id = secrets.token_hex(4)
user_folder = st.session_state.session_id
st.info(f"Your session ID: `{user_folder}`. Keep this to revisit your files (not secure).")

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
            st.error("Failed to fetch the article. Paste text instead.")

if article:
    st.caption(f"Article length: {len(article)} characters")

# Prediction button
if st.button("Run Entity Predictions"):
    if not filename_input:
        st.warning("Enter a filename before running predictions.")
    elif not article.strip():
        st.warning("Provide article text or valid URL.")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        article_id = f"{filename_input}_{timestamp}"

        # Lazy load models
        with st.spinner("Loading classification model..."):
            clf_pipeline = get_stage2_model()
        with st.spinner("Loading NER model..."):
            bert_model = get_ner_model()

        # Run stage 1
        with st.spinner("Analyzing entities..."):
            preds, count = predict_with_cached_model(article_id, bert_model, article, output_dir="article_predictions")
        st.success(f"Found {len(preds)} entities ({count} with specific roles)")

        # Prepare Stage 2 input
        convert_prediction_txt_to_csv(
            article_id=article_id,
            article=article,
            prediction_file=f"article_predictions/{article_id}_predictions.txt",
            article_text=article,
            output_csv="article_predictions/tc_input.csv"
        )
        input_df = pd.read_csv("article_predictions/tc_input.csv")
        new_stage2 = run_stage2_with_cached_model(article_id, clf_pipeline, input_df)
        combined = new_stage2  # existing merging logic can be added if needed
        combined.to_csv("article_predictions/tc_output.csv", index=False)

        # Display results
        with st.expander("Detected Entities", expanded=True):
            spans = bert_model.predict(article, return_format='spans')
            for s, e, role in [(int(x.split('\t')[2]), int(x.split('\t')[3]), x.split('\t')[4]) for x in preds]:
                text_seg = article[s:e].strip()
                conf = next((sp[f"prob_{role.lower()}"] for sp in spans if sp['start']==s and sp['end']==e), None)
                icon = '🟢' if role=='Protagonist' else '🔴' if role=='Antagonist' else '🔵' if role=='Innocent' else '⚪'
                st.markdown(f"{icon} **{text_seg}** - {role} (confidence: {conf:.3f}) [{s}-{e}]")

        with st.expander("Fine-Grained Role Predictions", expanded=True):
            for _, row in new_stage2.iterrows():
                roles = row['predicted_fine_margin']
                scores = row['predicted_fine_with_scores']
                formatted = ", ".join(f"{r}: {scores.get(r,0):.3f}" for r in roles)
                st.markdown(f"**{row['entity_mention']}** ({row['p_main_role']}): _{formatted}_")

st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team*")
