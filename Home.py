import streamlit as st
import pandas as pd
import altair as alt
import re
from sidebar import render_sidebar, ROLE_COLORS
from render_text import reformat_text_html_with_tooltips, predict_entity_framing, format_sentence_with_spans
from streamlit.components.v1 import html as st_html
import streamlit as st
import sys
import os
from pathlib import Path
from mode_tc_utils.preprocessing import convert_prediction_txt_to_csv
from mode_tc_utils.tc_inference import run_role_inference

# Add the seq directory to the path to import predict.py
sys.path.append(str(Path(__file__).parent / 'seq'))

try:
    from predict import predict_text_to_file
    PREDICTION_AVAILABLE = True
except ImportError as e:
    PREDICTION_AVAILABLE = False
    prediction_error = str(e)

#def generate_response(input_text):
    #model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
    #st.info(model.invoke(input_text))


# Narrative classification
def predict_narrative_classification(text, threshold=0.0):
    data = [
        {"narrative": "Western Interference", "confidence": 0.88},
        {"narrative": "Peace Sabotage",      "confidence": 0.85},
        {"narrative": "Proxy War",           "confidence": 0.80},
    ]
    df = pd.DataFrame(data)
    return df[df.confidence >= threshold]

# Free-form narrative extraction
def extract_narrative(text):
    return "Kremlin claims British pressure and additional Russian demands thwarted a potential Ukraine peace deal."

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


# --- Streamlit App ---

st.set_page_config(page_title="FRaN-X", initial_sidebar_state='expanded', layout="wide")
st.title("FRaN-X: Entity Framing & Narrative Analysis")

# Article input
st.header("1. Article Input")

article, labels, user_folder, threshold, role_filter, hide_repeat = render_sidebar()

# Allow users to edit the article text directly
article = st.text_area("Article", value=article if article else "", height=300, 
                      help="Paste or type your article text here. You can also load articles from the sidebar.")

# Debug info (can remove later)
if article:
    st.caption(f"📝 Article length: {len(article)} characters")

# Add prediction functionality right after the text area
if PREDICTION_AVAILABLE:
    st.success("🤖 **Entity Prediction Model Loaded**: Run predictions on any article text.")
    
    # Always show buttons if prediction is available
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Run Entity Predictions", help="Analyze entities in the current article", key="predict_main"):
            if article and article.strip():
                try:
                    with st.spinner("Analyzing entities in your article..."):
                        # Create output directory
                        predictions_dir = "article_predictions"
                        os.makedirs(predictions_dir, exist_ok=True)
                        
                        # Run prediction
                        predictions, non_unknown_count = predict_text_to_file(
                            text=article,
                            output_filename="current_article_predictions.txt",
                            output_dir=predictions_dir
                        )
                        # convert txt output of stage 1 into csv and prepare for text classification model 2
                        # also extracts context
                        stage2_csv_path = os.path.join(predictions_dir, "tc_input.csv")
                        convert_prediction_txt_to_csv(article,
                            prediction_file=os.path.join(predictions_dir, "current_article_predictions.txt"),
                            article_text=article,
                            output_csv=stage2_csv_path
                            )
                        
                        stage2_df = pd.read_csv("article_predictions/tc_input.csv")
                        stage2_df = run_role_inference(stage2_df)

                        output_path = os.path.join(predictions_dir, "tc_output.csv")
                        stage2_df.to_csv(output_path, index=False)
                    
                    st.success(f"✅ Entity analysis complete! Found {len(predictions)} entities ({non_unknown_count} with specific roles)")
                    
                    # Show detailed predictions
                    if predictions:
                        with st.expander("🎯 Detected Entities", expanded=True):
                            for i, pred in enumerate(predictions):
                                entity, start, end, role = pred.split('\t')
                                # Color code by role
                                if role == "Protagonist":
                                    st.markdown(f"🟢 **{entity}** - {role} (position {start}-{end})")
                                elif role == "Antagonist":
                                    st.markdown(f"🔴 **{entity}** - {role} (position {start}-{end})")
                                elif role == "Innocent":
                                    st.markdown(f"🔵 **{entity}** - {role} (position {start}-{end})")
                                else:
                                    st.markdown(f"⚪ **{entity}** - {role} (position {start}-{end})")

                    else:
                        st.info("No entities detected in the article.")

                    if not stage2_df.empty:
                        with st.expander("🧠 Fine-Grained Role Predictions", expanded=True):
                            for _, row in stage2_df.iterrows():
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
    
    with col2:
        if st.button("💾 Save Predictions to File", help="Save current predictions to txt_predictions folder", key="save_main"):
            if article and article.strip() and user_folder:
                try:
                    with st.spinner("Saving predictions..."):
                        # Create user-specific predictions directory
                        predictions_dir = os.path.join('txt_predictions', user_folder)
                        os.makedirs(predictions_dir, exist_ok=True)
                        
                        # Generate filename with timestamp
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"home_analysis_{timestamp}_predictions.txt"
                        
                        # Run prediction and save
                        predictions, non_unknown_count = predict_text_to_file(
                            text=article,
                            output_filename=filename,
                            output_dir=predictions_dir
                        )
                    
                    st.success(f"💾 Predictions saved to: txt_predictions/{user_folder}/{filename}")
                    st.info(f"📊 Summary: {len(predictions)} entities found ({non_unknown_count} with specific roles)")
                    
                except Exception as e:
                    st.error(f"Error saving predictions: {str(e)}")
            elif not article or not article.strip():
                st.warning("⚠️ Please enter some article text first.")
            elif not user_folder:
                st.warning("⚠️ Please select a user folder in the sidebar first.")
            else:
                st.warning("Entity prediction model is not available.")
else:
    st.warning(f"⚠️ **Entity Prediction Unavailable**: {prediction_error if 'prediction_error' in locals() else 'Model not loaded'}")

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
    st.header("7. Narrative Classification")
    df_n = predict_narrative_classification(article, threshold)
    st.dataframe(df_n)

    # 5. Free-form narrative extraction
    st.header("8. Free-form Narrative Extraction")
    st.write(extract_narrative(article))


    # 6. Bias identification & rewriting
    #st.header("6. Bias Identification & Rewriting")
    #mode_rw = st.radio("Rewrite mode", ["conservative","aggressive"])  
    #suggestions = identify_bias_and_rewrite(article, mode_rw)
    #for s in suggestions:
    #    st.write(f"**Span:** {s['span']}  \n**Suggestion:** {s['suggestion']}")

    #openai_api_key = st.text_input("OpenAI API Key", type="password")

    #st.write(role_sentences)


    #sent = st.selectbox("Choose sentence: ", role_sentences['sentence'])


    #text = "Enter text:" + "Give an explanation for why the following sentence has been annotated as an Antagonist, Protogonist, or Innocent based on the surrounding context:",sent
    
    #st.write(text)

    #submitted = st.form_submit_button("Submit")
    #if not openai_api_key.startswith("sk-"):
        #st.warning("Please enter your OpenAI API key!", icon="⚠")
    #if submitted and openai_api_key.startswith("sk-"):
        #generate_response(text)





st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")