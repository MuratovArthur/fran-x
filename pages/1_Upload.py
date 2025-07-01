import streamlit as st
import time
import requests
from bs4 import BeautifulSoup
from sidebar import render_sidebar
import os

st.set_page_config(page_title="FRaN-X", initial_sidebar_state='expanded', layout="wide")
st.title("Upload Articles")
st.write("Choose a User Folder in the Sidebar to continue")

# Add info about the workflow
st.info("📄 **Upload Workflow**: Save your articles here, then go to the Home page to run entity predictions and analysis.")

article, labels, user_folder, threshold, role_filter, hide_repeat = render_sidebar(True, False, True, True)

mode = st.radio("Input mode", ["Paste Text","URL"])
if mode == "Paste Text":
    article = st.text_area("Article text", height=200)
    os.makedirs("user_articles", exist_ok=True)

    filename_input = st.text_input("Filename (without extension)")

    # Save button
    if st.button("💾 Save Article"):
        if article.strip() and filename_input.strip():
            # Clean filename and enforce .txt extension
            safe_filename = filename_input.strip().replace(" ", "_") + ".txt"
            filepath = os.path.join('user_articles', user_folder, safe_filename)

            # Check for duplicate filenames
            if os.path.exists(filepath):
                st.error(f"A file named '{safe_filename}' already exists. Please choose a different name.")
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(article)
                st.success(f"✅ Article saved as {safe_filename}")
                st.info("💡 **Next Step**: Go to the Home page to run entity predictions and analysis on your article.")
                    
                time.sleep(1)  # Brief pause before rerun
                st.rerun()
        else:
            st.warning("Both article text and filename must be provided.")

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
                
                # Auto-generate filename from URL
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                auto_filename = parsed_url.netloc.replace("www.", "") + "_article"
                
                filename_input = st.text_input("Filename (without extension)", value=auto_filename)
                
                # Save button for URL content
                if st.button("💾 Save Article from URL"):
                    if filename_input.strip():
                        # Clean filename and enforce .txt extension
                        safe_filename = filename_input.strip().replace(" ", "_") + ".txt"
                        filepath = os.path.join('user_articles', user_folder, safe_filename)

                        # Check for duplicate filenames
                        if os.path.exists(filepath):
                            st.error(f"A file named '{safe_filename}' already exists. Please choose a different name.")
                        else:
                            with open(filepath, "w", encoding="utf-8") as f:
                                f.write(article)
                            st.success(f"✅ Article saved as {safe_filename}")
                            st.info("💡 **Next Step**: Go to the Home page to run entity predictions and analysis on your article.")
                                
                            time.sleep(1)  # Brief pause before rerun
                            st.rerun()
                    else:
                        st.warning("Filename must be provided.")
            else:
                st.warning("Could not extract meaningful content from the URL.")
        except Exception as e:
            st.error(f"Error fetching article from URL: {str(e)}")



st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")