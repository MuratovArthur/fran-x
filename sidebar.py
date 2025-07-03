import streamlit as st
import os
from load_annotations import load_article, load_labels_stage2

ROLE_COLORS = {
    "Protagonist": "#a1f4a1",
    "Antagonist":  "#f4a1a1",
    "Innocent":    "#a1c9f4",
}
def get_text_color():
    #bg_color = st.session_state.franx_theme.lower()
    #return "#000000" if bg_color in ("#ffffff", "ffffff") else "#ffffff"
    return '#000000'

def load_file_names(folder_path):
    files = os.listdir(folder_path)
    return tuple(files)

def render_sidebar(choose_user_folder = True, check_example = True, new_session = False, choose_user_file = True):
    user_folder = None

    st.sidebar.header("Settings")

    # State initialization
    if "use_example" not in st.session_state:
        st.session_state.use_example = False
    if "hide_repeat" not in st.session_state:
        st.session_state.hide_repeat = False
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.20
    if "role_filter" not in st.session_state:
        st.session_state.role_filter = list(ROLE_COLORS.keys())

    # Sidebar widget
    if check_example:
        use_example = st.sidebar.checkbox("Demo mode with example articles", value=st.session_state.use_example)
        if use_example != st.session_state.use_example:
            st.session_state.use_example = use_example
            st.rerun()
    else:
        use_example = False

    hide_repeat = st.sidebar.checkbox("Make repeat annotations transparent", value=st.session_state.hide_repeat)
    if hide_repeat != st.session_state.hide_repeat:
        st.session_state.hide_repeat = hide_repeat
        st.rerun()

    new_threshold = st.sidebar.slider("Narrative confidence threshold", 0.0, 1.0, st.session_state.threshold, 0.01)
    st.session_state.threshold = new_threshold

    role_filter = st.sidebar.multiselect(
        "Filter roles",
        options=list(ROLE_COLORS.keys()),
        default=st.session_state.role_filter
    )
    #if role_filter != st.session_state.role_filter:
        #st.session_state.role_filter = role_filter
        #st.rerun()

    article = ""
    labels = []

    folder_path = 'chunk_data' if use_example else 'user_articles'
    if choose_user_folder:
        if folder_path == "user_articles":
            if new_session:
                user_folder = st.sidebar.selectbox("User Folder", ["New Session"] + os.listdir(folder_path))
            else:
                user_folder = st.sidebar.selectbox("User Folder", os.listdir(folder_path))
                
            if user_folder == "New Session":
                user_number = str(int(st.sidebar.text_input("Input a number:", 0)))
                
                if user_number.strip():
                    user_folder = f"session_{user_number.strip()}"
                    full_user_path = os.path.join(folder_path, user_folder)
                    os.makedirs(full_user_path, exist_ok=True)
                    folder_path = full_user_path
                else:
                    st.stop()
            else:
                folder_path = os.path.join(folder_path, user_folder)

        if choose_user_file:
            file_names = load_file_names(folder_path)
            valid_files = [f for f in file_names if f and not f.startswith('.')]
            ##st.write(valid_files)
            file_options = ["Select a file"] + valid_files

            if "article_name" not in st.session_state or st.session_state.article_name not in valid_files:
                st.session_state.article_name = "Select a file"

            selected_index = file_options.index(st.session_state.article_name) if st.session_state.article_name in file_options else 0

            selected_file = st.sidebar.selectbox(
                "Choose a file",
                options=file_options,
                index=selected_index
            )

            if selected_file != st.session_state.article_name:
                st.session_state.article_name = selected_file
                st.rerun()

            if selected_file != "Select a file":
                file_path = os.path.join(folder_path, selected_file)
                article = load_article(file_path)
                labels = load_labels_stage2(selected_file, st.session_state.threshold)



            elif not valid_files:
                st.sidebar.warning("No files found in the selected folder.")

    return article, labels, user_folder, new_threshold, role_filter, hide_repeat
