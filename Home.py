import streamlit as st
import pandas as pd
import altair as alt
import requests
import datetime
import re
from sidebar import render_sidebar, ROLE_COLORS
from render_text import reformat_text_html_with_tooltips, predict_entity_framing, format_sentence_with_spans
from streamlit.components.v1 import html as st_html
import streamlit as st
import sys
import os
from pathlib import Path
import ast
from mode_tc_utils.preprocessing import convert_prediction_txt_to_csv
from mode_tc_utils.tc_inference import run_role_inference
from bs4 import BeautifulSoup
import secrets

# Add the seq directory to the path to import predict.py
sys.path.append(str(Path(__file__).parent / 'seq'))

# ============================================================================
# MODEL CACHING - Load both models once on app launch
# ============================================================================

@st.cache_resource
def load_ner_model():
    """Load the NER model once and cache it."""
    try:
        import torch
        from src.deberta import DebertaV3NerClassifier
        
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
            
        return bert_model
    except Exception as e:
        st.error(f"Failed to load NER model: {e}")
        return None



@st.cache_resource 
def load_stage2_model():
    """Load the stage 2 classification model once and cache it."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        
        model_path = "artur-muratov/franx-cls"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        clf_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        
        return clf_pipeline
    except Exception as e:
        st.error(f"Failed to load Stage 2 model: {e}")
        return None



def predict_with_cached_model(article_id, bert_model, text, output_filename="predictions.txt", output_dir="output"):
    """Run prediction using the cached NER model."""
    from pathlib import Path
    
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
        output_lines.append(f"{article_id}\t{entity_text}\t{s}\t{e}\t{role}")

    # Save predictions to txt file
    output_file_path = output_path / (article_id + "_predictions.txt")
    output_file_path.write_text('\n'.join(output_lines), encoding='utf-8')

    a = Path("article_predictions") / "current_article_preds.txt"
    a.write_text('\n'.join(output_lines), encoding='utf-8')
    
    return output_lines, non_unknown



def run_stage2_with_cached_model(article_id, clf_pipeline, df, threshold=0.01, margin=0.05):
    """Run stage 2 inference using the cached classification model."""

    def pipeline_with_confidence(example, threshold=threshold):
        input_text = (
            f"Entity: {example['entity_mention']}\n"
            f"Main Role: {example['p_main_role']}\n"
            f"Context: {example['context']}"
        )
        try:
            scores = clf_pipeline(input_text)[0]  # [{'label': ..., 'score': ...}]
        except Exception as e:
            print(f"Error in pipeline: {e}")
            return {}

        filtered_scores = {
            s['label']: round(s['score'], 4) for s in scores if s['score'] > threshold
        }
        return dict(sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True))

    def select_roles_within_margin(scores, margin=margin):
        if not scores:
            return []
        top_score = max(scores.values())
        return [role for role, score in scores.items() if score >= top_score - margin]

    def filter_scores_by_margin(row):
        scores = row['predicted_fine_with_scores']
        margin_roles = row['predicted_fine_margin']
        return {role: scores[role] for role in margin_roles if role in scores}

    # Apply predictions
    df['predicted_fine_with_scores'] = df.apply(pipeline_with_confidence, axis=1)
    df['predicted_fine_margin'] = df['predicted_fine_with_scores'].apply(select_roles_within_margin)
    df['p_fine_roles_w_conf'] = df.apply(filter_scores_by_margin, axis=1)
    df['article_id'] = article_id

    return df


# Load models on app startup
NER_MODEL = load_ner_model()
STAGE2_MODEL = load_stage2_model()

# Check if models loaded successfully
if NER_MODEL is not None and STAGE2_MODEL is not None:
    PREDICTION_AVAILABLE = True
    prediction_error = None
elif NER_MODEL is None:
    PREDICTION_AVAILABLE = False
    prediction_error = "NER model failed to load"
elif STAGE2_MODEL is None:
    PREDICTION_AVAILABLE = False
    prediction_error = "Stage 2 model failed to load"
else:
    PREDICTION_AVAILABLE = False
    prediction_error = "Both models failed to load"

#def generate_response(input_text):
    #model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
    #st.info(model.invoke(input_text))

def escape_entity(entity):
    return re.sub(r'([.^$*+?{}\[\]\\|()])', r'\\\1', entity)

def filter_labels_by_role(labels, role_filter):
    filtered = {}
    for entity, mentions in labels.items():
        filtered_mentions = [
            m for m in mentions if m.get("main_role") in role_filter
        ]
        if filtered_mentions:
            filtered[entity] = filtered_mentions
    return filtered


# ============================================================================
# STREAMLIT APP - Allow users to upload and save articles
# ============================================================================


st.set_page_config(page_title="FRaN-X", initial_sidebar_state='expanded', layout="wide")
st.title("FRaN-X: Entity Framing & Narrative Analysis")

#_, labels, user_folder, threshold, role_filter, hide_repeat = render_sidebar(False, False, False, False, False)
article = ""

def generate_unique_session_id(base_folder="user_articles", length=8):
    while True:
        session_id = secrets.token_hex(length // 2)
        session_folder = os.path.join(base_folder, session_id)
        if not os.path.exists(session_folder):
            return session_id


# Generate or retrieve a unique session ID for the user
if "session_id" not in st.session_state:
    st.session_state.session_id = generate_unique_session_id()
    #user_folder = st.session_state.session_id
    #st.info(f"Your session ID: `{user_folder}`\n\nNote this ID to return to your files later.\nℹ️ Your session ID keeps your work separate from others, but it is not secure. Do not upload sensitive or confidential information.")
    

user_folder = st.session_state.session_id



# Always show the session ID info
st.info(
    f"Your session ID: `{user_folder}`\n\n"
    "Note this ID to return to your files later.\n"
    "ℹ️ Your session ID keeps your work separate from others, but it is not secure. "
    "Do not upload sensitive or confidential information."
)
st.sidebar.info(f"Your session ID: `{user_folder}`")
# Article input
st.header("1. Article Input")

filename_input = st.text_input("Filename (without extension)")



mode = st.radio("Input mode", ["Paste Text","URL"])
if mode == "Paste Text":
    article = st.text_area("Article", value=article if article else "", height=300, help="Paste or type your article text here. You can also load articles from the sidebar.")    
    os.makedirs("user_articles", exist_ok=True)

else:
    url = st.text_input("Article URL")
    #article = ""
    #if url:
    #    try:
    #        with st.spinner("Fetching article from URL..."):
     #           resp = requests.get(url)
      #          soup = BeautifulSoup(resp.content, 'html.parser')
       #         article = '\n'.join(p.get_text() for p in soup.find_all('p'))
            
          #  if article.strip():
          #      st.text_area("Fetched Article", value=article, height=200, disabled=True)
          #  else:
          #      st.warning("Could not extract meaningful content from the URL. Please check a different URL or paste the text directly.")
        #except Exception as e:
         #   st.error("Sorry, we couldn't fetch or process the article from this URL. Please check that the link is correct and points to a public news article, or try pasting the text instead.")



# Debug info (can remove later)
if article:
    st.caption(f"Article length: {len(article)} characters")



# Add prediction functionality right after the text area
if PREDICTION_AVAILABLE:
    st.success("🤖 **Both Models Loaded**: Ready for entity prediction and fine-grained role classification.")
    filename = ""
    predictions_dir = ""
    # Always show buttons if prediction is available
    
    if st.button("Run Entity Predictions", help="Analyze entities in the current article", key="predict_main"):

        if mode == "URL":
            if not url or not url.strip():
                st.warning("⚠️ Please enter a valid URL before running predictions.")
                st.stop()
            try:
                with st.spinner("Fetching article from URL..."):
                    resp = requests.get(url)
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    article = '\n'.join(p.get_text() for p in soup.find_all('p'))
                if not article.strip():
                    st.warning("Could not extract meaningful content from the URL. Please check a different URL or paste the text directly.")
                    st.stop()
            except Exception:
                st.error("Sorry, we couldn't fetch or process the article from this URL. Please check that the link is correct and points to a public news article, or try pasting the text instead.")
                st.stop()

        elif mode == "Paste Text":

            if not filename_input:
                st.warning("⚠️ Please enter a filename for the article before running predictions.")
                st.stop()
        # Generate filename

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #filename = f"{filename_input}_{timestamp}_predictions.csv"


        #filename_wo_pred = f"{filename_input}_{timestamp}"
        #a = Path("user_articles") / user_folder / filename_wo_pred
        #a.write_text(article, encoding='utf-8')

        filename_wo_pred = f"{filename_input}_{timestamp}"
        user_dir = Path("user_articles") / user_folder
        user_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        


        if article and article.strip():
            try:
                with st.spinner("Analyzing entities in your article..."):
                    # Create output directory
                    predictions_dir = "article_predictions"
                    os.makedirs(predictions_dir, exist_ok=True)
                        
                    # Run prediction with cached NER model
                    #puts values in the current_articles_predictions.txt file
                    predictions, non_unknown_count = predict_with_cached_model(
                        article_id=filename_wo_pred,
                        bert_model=NER_MODEL,
                        text=article,
                        output_filename="current_article_preds.txt",
                        output_dir=predictions_dir
                    )

                
                        
                    # convert txt output of stage 1 into csv and prepare for text classification model 2
                    # also extracts context
                    #puts things into tc_input
                    # Step 1: Load existing tc_output.csv (if it exists)
                    input_stage2_csv_path = os.path.join(predictions_dir, "tc_input.csv")
                    output_stage2_csv_path = os.path.join(predictions_dir, "tc_output.csv")

                    if os.path.exists(output_stage2_csv_path):
                        existing_df = pd.read_csv(output_stage2_csv_path)
                    else:
                        existing_df = pd.DataFrame()

                    # Step 2: Convert Stage 1 predictions into CSV
                    convert_prediction_txt_to_csv(
                        article_id=filename_wo_pred,
                        article=article,
                        prediction_file=os.path.join(predictions_dir, "current_article_preds.txt"),
                        article_text=article,
                        output_csv=input_stage2_csv_path
                    )

                    # Step 3: Load newly written tc_input.csv
                    new_input_df = pd.read_csv(input_stage2_csv_path)

                    # Step 4: Run Stage 2 predictions on new inputs
                    new_stage2_df = run_stage2_with_cached_model(filename_wo_pred, STAGE2_MODEL, new_input_df)

                    # Step 5: Merge existing + new predictions
                    combined_df = pd.concat([existing_df, new_stage2_df], ignore_index=True)

                    # Step 6: Save to tc_output.csv
                    output_path = os.path.join(predictions_dir, "tc_output.csv")
                    combined_df.to_csv(output_path, index=False, encoding="utf-8")

                    #st.success(f"✅ tc_output.csv updated with {len(new_stage2_df)} new rows ({len(combined_df)} total)")
                    
                    st.success(f"✅ Entity analysis complete! Found {len(predictions)} entities ({non_unknown_count} with specific roles)")
                    
                    # Show detailed predictions with confidence scores
                    if predictions:
                        a = user_dir / filename_wo_pred
                        a.write_text(article, encoding='utf-8')
                        with st.expander("🎯 Detected Entities", expanded=True):
                            # Get all entity spans with confidence scores ONCE (not in the loop!)
                            entity_spans = NER_MODEL.predict(article, return_format='spans')
                                
                            for i, pred in enumerate(predictions):
                                text_id, entity, start, end, role = pred.split('\t')
                                    
                                # Find matching span for this entity
                                confidence_score = None
                                for span in entity_spans:
                                    if span['start'] == int(start) and span['end'] == int(end):
                                        if role == "Protagonist":
                                            confidence_score = span['prob_protagonist']
                                        elif role == "Antagonist":
                                            confidence_score = span['prob_antagonist']
                                        elif role == "Innocent":
                                            confidence_score = span['prob_innocent']
                                        elif role == "Unknown":
                                            confidence_score = span['prob_unknown']
                                        break
                                    
                                confidence_text = f" (confidence: {confidence_score:.3f})" if confidence_score is not None else ""
                                    
                                # Color code by role
                                if role == "Protagonist":
                                    st.markdown(f"🟢 **{entity}** - {role}{confidence_text} (position {start}-{end})")
                                elif role == "Antagonist":
                                    st.markdown(f"🔴 **{entity}** - {role}{confidence_text} (position {start}-{end})")
                                elif role == "Innocent":
                                    st.markdown(f"🔵 **{entity}** - {role}{confidence_text} (position {start}-{end})")
                                else:
                                    st.markdown(f"⚪ **{entity}** - {role}{confidence_text} (position {start}-{end})")

                    else:
                        st.info("No entities detected in the article.")

                    if not new_stage2_df.empty:
                        with st.expander("🧠 Fine-Grained Role Predictions", expanded=True):
                            for _, row in new_stage2_df.iterrows():
                                entity = row.get("entity_mention", "N/A")
                                main_role = row.get("p_main_role", "N/A")

                                # Parse list of fine roles and their scores
                                fine_roles = row.get("predicted_fine_margin", [])
                                fine_scores = row.get("predicted_fine_with_scores", {})

                                if isinstance(fine_roles, str):
                                    try:
                                        fine_roles = ast.literal_eval(fine_roles)
                                    except:
                                        fine_roles = []

                                if isinstance(fine_scores, str):
                                    try:
                                        fine_scores = ast.literal_eval(fine_scores)
                                    except:
                                        fine_scores = {}

                                # Format role + score for display
                                formatted_roles = ", ".join(
                                f"{role}: confidence = {fine_scores.get(role, '—')}" for role in fine_roles
                                    ) if fine_roles else "None"


                                st.markdown(f"**{entity}** ({main_role}): _{formatted_roles}_")

                        #st.write("Stored results. Head to the other pages for further analysis.")
                        st.info("✅ Results stored successfully. Explore the other pages to dive deeper into the analysis.")

            except Exception as e:
                 st.error("No entities found in the article. Please upload a longer or different article.")
        else:
            if not article or not article.strip():
                if mode == "Paste Text":
                    st.warning("⚠️ Please enter some article text first.")
                
                else:
                    st.warning("⚠️ Please enter a valid URL or paste article as text.")
                    #else:
            #st.warning("⚠️ Please enter some article text first.")

            
        st.markdown("---")
    
    #with col2:
    #    if st.button("💾 Save Predictions to File", help="Save current predictions to txt_predictions folder", key="save_main"):
    #        if article and article.strip() and user_folder:
    #            try:
    #                with st.spinner("Saving predictions..."):
    #                    # Create user-specific predictions directory
    #                    predictions_dir = os.path.join('txt_predictions', user_folder)
    #                    os.makedirs(predictions_dir, exist_ok=True)
    #                    
    #                    # Run prediction with cached model and save
    #                    predictions, non_unknown_count = predict_with_cached_model(
    #                        article_id=filename,
    #                        bert_model=NER_MODEL,
    #                        text=article,
    #                        output_filename=filename,
    #                        output_dir=predictions_dir
    #                    )
    #                
    #                st.success(f"💾 Predictions saved to: txt_predictions/{user_folder}/{filename}")
    #                st.info(f"📊 Summary: {len(predictions)} entities found ({non_unknown_count} with specific roles)")
    #                
    #            except Exception as e:
    #                st.error(f"Error saving predictions: {str(e)}")
    #        elif not article or not article.strip():
    #            st.warning("⚠️ Please enter some article text first.")
    #        elif not user_folder:
    #            st.warning("⚠️ Please select a user folder in the sidebar first.")
    #        else:
    #            st.warning("Entity prediction model is not available.")
else:
    st.warning(f"⚠️ **Entity Prediction Unavailable**: {prediction_error if prediction_error else 'Models not loaded'}")





st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")