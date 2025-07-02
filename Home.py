import streamlit as st
import pandas as pd
import altair as alt
import requests
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
        #st.write(f"article_ID{article_id}")
        output_lines.append(f"{article_id}\t{entity_text}\t{s}\t{e}\t{role}")

    # Save predictions to txt file
    output_file_path = output_path / (article_id + "_predictions.txt")
    ##st.write(output_file_path)
    ##st.write(user_folder)
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

_, labels, user_folder, threshold, role_filter, hide_repeat = render_sidebar(True, False, True, True)
article = ""

# Article input
st.header("1. Article Input")

filename_input = st.text_input("Filename (without extension)")

mode = st.radio("Input mode", ["Paste Text","URL"])
if mode == "Paste Text":
    article = st.text_area("Article", value=article if article else "", height=300, help="Paste or type your article text here. You can also load articles from the sidebar.")    
    os.makedirs("user_articles", exist_ok=True)

else:
    url = st.text_input("Article URL")
    article = ""
    if url:
        try:
            with st.spinner("Fetching article from URL..."):
                resp = requests.get(url)
                soup = BeautifulSoup(resp.content, 'html.parser')
                article = '\n'.join(p.get_text() for p in soup.find_all('p'))
            
            if article.strip():
                st.text_area("Fetched Article", value=article, height=200, disabled=True)
            else:
                st.warning("Could not extract meaningful content from the URL.")
        except Exception as e:
            st.error(f"Error fetching article from URL: {str(e)}")



# Debug info (can remove later)
if article:
    st.caption(f"Article length: {len(article)} characters")



# Add prediction functionality right after the text area
if PREDICTION_AVAILABLE:
    st.success("🤖 **Both Models Loaded**: Ready for entity prediction and fine-grained role classification.")
    filename = ""
    predictions_dir = ""
    # Always show buttons if prediction is available
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Run Entity Predictions", help="Analyze entities in the current article", key="predict_main"):
            # Generate filename
            import datetime
            from pathlib import Path

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_input}_{timestamp}_predictions.csv"
            ##st.write(filename)


            filename_wo_pred = f"{filename_input}_{timestamp}"
            a = Path("user_articles") / user_folder / filename_wo_pred
            a.write_text(article, encoding='utf-8')



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

                        st.success(f"✅ tc_output.csv updated with {len(new_stage2_df)} new rows ({len(combined_df)} total)")
                    
                    st.success(f"✅ Entity analysis complete! Found {len(predictions)} entities ({non_unknown_count} with specific roles)")
                    
                    # Show detailed predictions with confidence scores
                    if predictions:
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
                                f"{role}: confidence = ({fine_scores.get(role, '—')})" for role in fine_roles
                                    ) if fine_roles else "None"


                                st.markdown(f"**{entity}** ({main_role}): _{formatted_roles}_")


                        
                except Exception as e:
                    st.error(f"Error running entity prediction: {str(e)}")
            else:
                st.warning("⚠️ Please enter some article text first.")
    
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









if article and labels:
    show_annot   = st.checkbox("Show annotated article view", True)
    df_f = predict_entity_framing(labels, threshold)

    # 2. Annotated article view
    if show_annot:
        st.header("3. Annotated Article")
        html = reformat_text_html_with_tooltips(article, filter_labels_by_role(labels, role_filter), hide_repeat)
        st.components.v1.html(html, height=600, scrolling = True)     

    # 3. Entity framing & timeline

    if not df_f.empty:
        df_f = df_f[df_f['main_role'].isin(role_filter)]

        st.header("4. Role Distribution & Transition Timeline")
        dist = df_f['main_role'].value_counts().reset_index()
        dist.columns = ['role','count']
        
        color_list = [ROLE_COLORS.get(role, "#cccccc") for role in dist['role']]
        domain_list = dist['role'].tolist()

        #chart        
        exploded = df_f.explode('fine_roles')
        grouped = exploded.groupby(['main_role', 'fine_roles']).size().reset_index(name='count')
        grouped = grouped.sort_values(by=['main_role', 'fine_roles'])

        # Compute the cumulative sum within each main_role
        grouped['cumsum'] = grouped.groupby('main_role')['count'].cumsum()
        grouped['prevsum'] = grouped['cumsum'] - grouped['count']
        grouped['entities'] = grouped['prevsum'] + grouped['count'] / 2

        # Bar chart
        bars = alt.Chart(grouped).mark_bar(stroke='black', strokeWidth=0.5).encode(
            x=alt.X('main_role:N', title='Main Role'),
            y=alt.Y('count:Q', stack='zero'),
            color=alt.Color('main_role:N', scale=alt.Scale(domain=domain_list, range=color_list), legend=None),
            tooltip=['main_role', 'fine_roles', 'count']
        )

        label_chart = alt.Chart(grouped).mark_text(
            color='black',
            fontSize=11
        ).encode(
            x='main_role:N',
            y=alt.Y('entities:Q'),  # <- exact center of the segment
            text='fine_roles:N'
        )

        # Combine
        chart = (bars + label_chart).properties(
            width=500,
            title='Main Roles with Fine-Grained Role Segments'
        )

        st.altair_chart(chart, use_container_width=True)

        #timeline
        timeline = alt.Chart(df_f).mark_bar().encode(
            x=alt.X('start:Q', title='Position'), x2='end:Q',
            y=alt.Y('entity:N', title='Entity'),
            color=alt.Color('main_role:N', scale=alt.Scale(domain=list(ROLE_COLORS.keys()), range=list(ROLE_COLORS.values()))),
            tooltip=['entity','main_role','confidence']
        ).properties(height=200)
        st.altair_chart(timeline, use_container_width=True)

        role_counts = df_f['main_role'].value_counts().reset_index()
        role_counts.columns = ['main_role', 'count']

        #pie chart
        pie = alt.Chart(role_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field='count', type='quantitative'),
            color=alt.Color(field='main_role', type='nominal', scale=alt.Scale(domain=list(ROLE_COLORS.keys()), range=list(ROLE_COLORS.values()))),
            tooltip=['main_role', 'count']
        ).properties(title="Main Role Distribution")

        st.altair_chart(pie, use_container_width=True)

    # --- Sentence Display by Role with Adaptive Layout ---
    st.markdown("## 5. Sentences by Role Classification")

    df_f['main_role'] = df_f['main_role'].str.strip().str.title()
    df_f['fine_roles'] = df_f['fine_roles'].apply(lambda roles: [r.strip().title() for r in roles if r.strip()])
    df_f = df_f[df_f['main_role'].isin(ROLE_COLORS)]

    main_roles = sorted(df_f['main_role'].unique())
    multiple_roles = len(main_roles) > 1
    cols_per_row = 2 if multiple_roles else 1

    role_cols = [st.columns(cols_per_row) for _ in range((len(main_roles) + cols_per_row - 1) // cols_per_row)]

    for idx, role in enumerate(main_roles):
        col = role_cols[idx // cols_per_row][idx % cols_per_row]

        with col:
            role_df = df_f[df_f['main_role'] == role][['sentence', 'fine_roles']].copy()
            role_df['fine_roles'] = role_df['fine_roles'].apply(tuple)
            role_sentences = role_df.drop_duplicates()

            st.markdown(
                f"<div style='background-color:{ROLE_COLORS[role]}; "
                f"padding: 8px; border-radius: 6px; font-weight:bold;'>"
                f"{role} — {len(role_sentences)} labels"
                f"</div>",
                unsafe_allow_html=True
            )
            
            seen_fine_roles = None
            for sent in role_sentences['sentence'].unique():
                html_block, seen_fine_roles = format_sentence_with_spans(sent, filter_labels_by_role(labels, role_filter), threshold, hide_repeat, False, seen_fine_roles)
                st.markdown(html_block, unsafe_allow_html = True)

            fine_df = df_f[df_f['main_role'] == role].explode('fine_roles')
            fine_df = fine_df[fine_df['fine_roles'].notnull() & (fine_df['fine_roles'] != '')]
            fine_roles = sorted(fine_df['fine_roles'].dropna().unique())

            if fine_roles and len(fine_roles)>1:
                selected_fine = st.selectbox(
                    f"Filter {role} by fine-grained role:",
                    ["Show all"] + fine_roles,
                    key=f"fine_{role}"
                )

                if selected_fine != "Show all":
                    fine_sents = fine_df[fine_df['fine_roles'] == selected_fine]['sentence'].drop_duplicates()
                    st.markdown(f"**{selected_fine}** — {len(fine_sents)} sentence(s):")
                    seen_fine_roles = None
                    for s in fine_sents:
                        html_block,seen_fine_roles = format_sentence_with_spans(s, filter_labels_by_role(labels, role_filter), threshold, hide_repeat, True, seen_fine_roles)
                        st_html(html_block, height=80, scrolling=False)
            else: 
                for fine_role in fine_roles:
                    st.write(f"All annotations of this main role are of type: {fine_role}")

    # Confidence Distribution
    #tweak details once confidence column uses real data
    st.subheader("6. Histogram of Confidence Levels")

    chart = alt.Chart(df_f).mark_bar().encode(
        alt.X("confidence:Q", bin=alt.Bin(maxbins=20), title="Confidence"),
        alt.Y("count()", title="Frequency"),
        tooltip=['count()']
    ).properties(
        width=50,
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    # 4. Narrative classification
    #st.header("7. Narrative Classification")
    #df_n = predict_narrative_classification(article, threshold)
    #st.dataframe(df_n)

    # 5. Free-form narrative extraction
    #st.header("8. Free-form Narrative Extraction")
    #st.write(extract_narrative(article))


    # 6. Bias identification & rewriting
    #st.header("6. Bias Identification & Rewriting")
    #mode_rw = st.radio("Rewrite mode", ["conservative","aggressive"])  
    #suggestions = identify_bias_and_rewrite(article, mode_rw)
    #for s in suggestions:
    #    st.write(f"**Span:** {s['span']}  \n**Suggestion:** {s['suggestion']}")

    #openai_api_key = st.text_input("OpenAI API Key", type="password")

    ##st.write(role_sentences)


    #sent = st.selectbox("Choose sentence: ", role_sentences['sentence'])


    #text = "Enter text:" + "Give an explanation for why the following sentence has been annotated as an Antagonist, Protogonist, or Innocent based on the surrounding context:",sent
    
    ##st.write(text)

    #submitted = st.form_submit_button("Submit")
    #if not openai_api_key.startswith("sk-"):
        #st.warning("Please enter your OpenAI API key!", icon="⚠")
    #if submitted and openai_api_key.startswith("sk-"):
        #generate_response(text)





st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")