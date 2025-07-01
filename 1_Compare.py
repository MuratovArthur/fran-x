import streamlit as st
from sidebar import render_sidebar
from load_annotations import load_article, load_labels, load_file_names
from Home import reformat_text_html_with_tooltips

ROLE_COLORS = {
    "Protagonist": "#a1f4a1",
    "Antagonist":  "#f4a1a1",
    "Innocent":    "#a1c9f4",
}

st.set_page_config(page_title="Compare", page_icon="📈", layout="wide")

article, labels, use_example, threshold, role_filter = render_sidebar(False)

st.markdown("# Compare Articles")

st.write(
    """Select a second article and compare role distribution between them"""
)

use_example = st.session_state.get("use_example", False)
threshold = st.session_state.get("threshold", 0.5)
article_folder_path = 'chunk_data' if use_example else 'user_articles'
label_folder_path = 'split_data' if use_example else 'user_labels'
file_names = [f for f in load_file_names(article_folder_path) if f and not f.startswith('.')]
file_options = ["Select a file"] + file_names

col1, col2 = st.columns(2)

with col1:
    file1 = st.selectbox("Select article (left)", file_options, key="left_article")
    if file1 != "Select a file":
        article1 = load_article(f"{article_folder_path}/{file1}")
        st.subheader(f"Article: {file1}")
        st.text_area("Content", article1, height=300, key = "left_text")

        labels1 = load_labels(label_folder_path, file1, threshold)

        html1 = reformat_text_html_with_tooltips(article1, labels1)
        st.components.v1.html(html1, height=600) 


with col2:
    file2 = st.selectbox("Select article (right)", file_options, key="right_article")
    if file2 != "Select a file":
        article2 = load_article(f"{article_folder_path}/{file2}")
        st.subheader(f"Article: {file2}")
        st.text_area("Content", article2, height=300, key = "right_text")

        labels2 = load_labels(label_folder_path, file2, threshold)

        html2 = reformat_text_html_with_tooltips(article2, labels2)
        st.components.v1.html(html2, height=1000) 