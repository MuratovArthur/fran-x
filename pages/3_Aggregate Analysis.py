import streamlit as st
import pandas as pd
import os
import itertools
import tempfile
import networkx as nx
import streamlit.components.v1 as components
from collections import Counter
from pyvis.network import Network
from sidebar import render_sidebar, load_file_names, ROLE_COLORS
from load_annotations import load_article, load_labels_stage2
from render_text import predict_entity_framing, normalize_entities


st.set_page_config(page_title="FRaN-X", initial_sidebar_state='expanded', layout="wide")
st.title("Aggregate Visualization")

article, labels, user_folder, threshold, role_filter, hide_repeat = render_sidebar(True, True, False, False)

folder_path = 'chunk_data' if user_folder == None else 'user_articles/'+user_folder

# --- Multiselect with rerun trigger ---
file_options = list(load_file_names(folder_path))
if "selected_files" not in st.session_state:
    st.session_state.selected_files = []

selected = st.sidebar.multiselect(
    "File(s)",
    options=file_options
)

if set(selected) != set(st.session_state.selected_files):
    st.session_state.selected_files = selected
    st.rerun()

files = st.session_state.selected_files

network_rows = []

for f in files:
    article_text = load_article(f'{folder_path}/{f}').strip()
    labels = load_labels_stage2(f, threshold)
    df_network = predict_entity_framing(labels, threshold)
    df_network = df_network[df_network['main_role'].isin(role_filter)]
    df_network = df_network.explode("fine_roles")
    df_network['fine_roles'] = df_network['fine_roles'].str.strip().str.title()
    df_network = df_network[df_network['fine_roles'].notnull() & (df_network['fine_roles'] != '')]
    df_network['article'] = article_text
    network_rows.append(df_network)

graph_df = pd.DataFrame()
if network_rows:
    graph_df = pd.concat(network_rows)

    # Create node labels
    graph_df["node_label"] = graph_df["entity"] + " (" + graph_df["fine_roles"] + ")"

    # Count mentions for node sizes
    node_sizes = graph_df["node_label"].value_counts().to_dict()

    # Assign colors based on main_role
    main_roles = graph_df["main_role"].unique()

    co_occurrence = Counter()
    for _, doc_df in graph_df.groupby("article"):
        nodes = doc_df["node_label"].unique()
        pairs = itertools.combinations(sorted(nodes), 2)
        co_occurrence.update(pairs)

    graph_df = normalize_entities(graph_df, 70)

# --- Interactive Network Graph ---
if not graph_df.empty:
    st.header("Network Graph")
    st.write("Interact with the connection between the entity,role nodes present across multiple documents. The edges represent a document where the two nodes are both present. ")
    st.write("You may have to wait a moment after changing selections for the graph to display. ")

    restrict_entities = st.multiselect("Entities", options=graph_df['entity'].unique(), default=graph_df['entity'].unique())
    graph_df = graph_df[graph_df['entity'].isin(restrict_entities)]

    # Build the graph
    G = nx.Graph()
    for _, row in graph_df.iterrows():
        node_id = f"{row['entity']} → {row['fine_roles']}"
        role = row['main_role']
        color = ROLE_COLORS.get(role, "#cccccc")

        if G.has_node(node_id):
            G.nodes[node_id]['size'] += 5
        else:
            G.add_node(node_id, label=node_id, color=color, size=15)

    for _, group in graph_df.groupby('article'):
        entities = group.apply(lambda r: f"{r['entity']} → {r['fine_roles']}", axis=1).unique()
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]
                if G.has_edge(e1, e2):
                    G[e1][e2]['weight'] += 5
                else:
                    G.add_edge(e1, e2, weight=0.5)

    # Create and render the network
    net = Network(height="1000px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)
    net.repulsion(node_distance=150, spring_length=200)
    net.show_buttons(filter_=["physics"])

    # Use temporary file properly
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        tmp_path = tmp_file.name

    with open(tmp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=1500, scrolling=True)

    # Optional cleanup
    os.remove(tmp_path)


st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")