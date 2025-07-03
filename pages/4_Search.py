import streamlit as st
import os
from sidebar import render_sidebar, load_file_names
from load_annotations import load_article, load_labels_stage2
from render_text import reformat_text_html_with_tooltips

st.set_page_config(page_title="FRaN-X", initial_sidebar_state='expanded', layout="wide")
st.title("Search")
st.write("Select File(s) in the sidebar to get started")

article, labels, user_folder, threshold, role_filter, hide_repeat = render_sidebar(True, True, False, False)

folder_path = 'chunk_data' if user_folder == None else 'user_articles'


if user_folder != None:
    #user_folder = st.sidebar.selectbox("", os.listdir(folder_path))
    folder_path = os.path.join(folder_path, user_folder)
files = st.sidebar.multiselect(
    "File(s)",
    options=list(load_file_names(folder_path))
)

word = st.text_input("Search for: ")

for f in files:
    article = load_article(f'{folder_path}/{f}').strip()
    if word.lower() in article.lower():
        labels = load_labels_stage2(
            f,
            threshold
        )

        st.write(f"#### {f} \n")

        with st.expander("Article", expanded=False):
            html = reformat_text_html_with_tooltips(article, labels, hide_repeat, word)
            st.components.v1.html(html, height=600, scrolling = True)     


st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")