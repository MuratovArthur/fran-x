import streamlit as st
import pandas as pd
import altair as alt
import colorsys
import tempfile
import os
import networkx as nx
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import matplotlib.colors as mcolors
from streamlit_echarts import st_echarts
from pyvis.network import Network
from collections import Counter
import itertools
from sidebar import render_sidebar, load_file_names, ROLE_COLORS
from load_annotations import load_article, load_labels_stage2
from render_text import reformat_text_html_with_tooltips, predict_entity_framing, normalize_entities

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

def rgb_to_hex(rgb_tuple):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb_tuple[0]*255), int(rgb_tuple[1]*255), int(rgb_tuple[2]*255)
    )

def generate_shades(base_hex, n):
    base_rgb = hex_to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    return [
        rgb_to_hex(colorsys.hls_to_rgb(h, max(0, min(1, l + i * 0.04)), s))
        for i in range(n)
    ]


st.set_page_config(page_title="FRaN-X", layout="wide")

# Sidebar
article, labels, user_folder, threshold, role_filter, hide_repeat = render_sidebar(False)

# Title Row with Dynamic Column Buttons
title_col, spacer, add_col, remove_col = st.columns([3, 5, 1, 1])
with title_col:
    st.markdown("# Compare Articles")
with add_col:
    if st.button("➕", use_container_width=True):
        st.session_state.column_count = min(st.session_state.get("column_count", 1) + 1, 4)
with remove_col:
    if st.button("➖", use_container_width=True):
        st.session_state.column_count = max(st.session_state.get("column_count", 1) - 1, 1)

st.write("Select one or more articles to compare their role distributions and contents side-by-side.")

# Initialize column count
column_count = st.session_state.get("column_count", 1)

# Prepare file options
use_example = st.session_state.get("use_example", False)
threshold = st.session_state.get("threshold", 0.5)
article_folder = 'chunk_data' if use_example else 'user_articles'
label_folder = 'article_predictions'

if user_folder == None:
    article_folder = 'chunk_data'
else:
    session_names = [f for f in load_file_names('user_articles') if f and not f.startswith('.')]
    user_folder = st.selectbox("Session Number", session_names)
    article_folder = os.path.join('user_articles', user_folder)



file_names = [f for f in load_file_names(article_folder) if f and not f.startswith('.')]
##st.write(file_names)
file_options = ["Select a file"] + file_names

# Store role distributions for each article
distribution_data = []

# Render dynamic columns
columns = st.columns(column_count)
for i, col in enumerate(columns):
    with col:
        st.markdown(f"#### Article {i + 1}")
        selected_file = st.selectbox(f"Choose a file ({i + 1})", file_options, key=f"file_{i}")
        
        if selected_file != "Select a file":
            article_text = load_article(f"{article_folder}/{selected_file}")
            labels = load_labels_stage2(selected_file, threshold)
            

            html = reformat_text_html_with_tooltips(article_text, labels, hide_repeat)
            line_count = article_text.count("\n") + 1
            st.components.v1.html(html, height=800, scrolling = True)

            # Role distribution collection
            df_f = predict_entity_framing(labels, threshold)

            df_f = df_f[df_f['main_role'].isin(role_filter)]
            if not df_f.empty:
                counts = df_f['main_role'].value_counts().reset_index()
                counts.columns = ['main_role', 'count']
                counts['article'] = selected_file
                distribution_data.append(counts)


# Sidebar Toggles
st.sidebar.header(" Data Visualization")
check_stacked = st.sidebar.checkbox("100% Stacked Bar: Main Role Distribution Comparision", True)
check_bar = st.sidebar.checkbox("Bar Chart: Fine Grain by Main Role Breakdown", True)
check_pie_main = st.sidebar.checkbox("Pie Chart: Fine Grain by Main Role Breakdown", True)
check_pie_role = st.sidebar.checkbox("Pie Chart: Role Breakdown", True)
check_network_static = st.sidebar.checkbox("Network Graph: Static", True)
check_network_interactive = st.sidebar.checkbox("Netowrk Graph: Interactive", True)


# Compare Distributions Across Articles
if distribution_data:

    # --- 100% Stacked Bar: Main Role Comparision ---

    if check_stacked:
        st.markdown("## Main Role Distribution by File Comparison")

        # Combine all role counts
        combined_df = pd.concat(distribution_data)

        # Compute percentages for 100% stacked chart
        total_per_article = combined_df.groupby('article')['count'].transform('sum')
        combined_df['percentage'] = combined_df['count'] / total_per_article * 100

        # Sort roles and articles for consistent color/domain
        domain_list = list(ROLE_COLORS.keys())
        color_list = [ROLE_COLORS.get(role, "#cccccc") for role in domain_list]

        # Altair stacked horizontal bar chart
        chart = alt.Chart(combined_df).mark_bar().encode(
            y=alt.Y('article:N', title="Article", sort=None),
            x=alt.X('percentage:Q', stack='normalize', title='Role Composition (%)'),
            color=alt.Color('main_role:N', scale=alt.Scale(domain=domain_list, range=color_list), legend=alt.Legend(title="Main Role")),
            tooltip=['article', 'main_role', 'count', alt.Tooltip('percentage:Q', format=".1f")]
        ).properties(
            width=700,
            height=200 * len(combined_df['article'].unique())
        )

        st.altair_chart(chart, use_container_width=True)

    # --- Cumulative Pie Chart by Main Role → Fine-Grained Role ---

    # Reload detailed entity data for pie chart
    all_detailed_data = []
    for i, col in enumerate(columns):
        selected_file = st.session_state.get(f"file_{i}")
        if selected_file and selected_file != "Select a file":
            article_text = load_article(f"{article_folder}/{selected_file}")
            labels = load_labels_stage2(selected_file, threshold)
            df_detail = predict_entity_framing(labels, threshold)
            df_detail = df_detail[df_detail['main_role'].isin(role_filter)]
            all_detailed_data.append(df_detail)

    if all_detailed_data:
        cumulative_df = pd.concat(all_detailed_data)

        # Explode fine_roles
        cumulative_df = cumulative_df.explode("fine_roles")
        cumulative_df['fine_roles'] = cumulative_df['fine_roles'].str.strip().str.title()
        cumulative_df = cumulative_df[cumulative_df['fine_roles'].notnull() & (cumulative_df['fine_roles'] != '')]

        # Group by both main_role and fine_role
        role_fine_counts = cumulative_df.groupby(['main_role', 'fine_roles']).size().reset_index(name='count')

            # --- Bar Chart ---
        if check_bar:
            #df_f = df_f[df_f['main_role'].isin(role_filter)]

            st.header("Bar: Cumulative Fine Grain by Main Role Breakdown")
            dist = cumulative_df['main_role'].value_counts().reset_index()
            dist.columns = ['role','count']
            
            color_list = [ROLE_COLORS.get(role, "#cccccc") for role in dist['role']]
            domain_list = dist['role'].tolist()

            #chart        
            #exploded = df_f.explode('fine_roles')
            grouped = cumulative_df.groupby(['main_role', 'fine_roles']).size().reset_index(name='count')
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


        if check_pie_main:
            st.header("Pie: Cumulative Fine Grain by Main Role Breakdown")

            # Assign colors
            fine_role_colors = {}
            for role in role_fine_counts['main_role'].unique():
                fine_roles = role_fine_counts[role_fine_counts['main_role'] == role]['fine_roles'].unique()
                base_color = ROLE_COLORS.get(role, "#cccccc")
                shades = generate_shades(base_color, len(fine_roles))
                for fine, shade in zip(fine_roles, shades):
                    fine_role_colors[fine] = shade

            color_scale = alt.Scale(domain=list(fine_role_colors.keys()), range=list(fine_role_colors.values()))

            # Percentages
            role_fine_counts['percentage'] = (
                role_fine_counts.groupby('main_role')['count']
                .transform(lambda x: x / x.sum() * 100)
            )

            # Base chart with NO legend
            base_no_legend = alt.Chart(role_fine_counts).mark_arc(innerRadius=30).encode(
                theta=alt.Theta(field='count', type='quantitative'),
                color=alt.Color('fine_roles:N', scale=color_scale, legend=None),
                tooltip=['main_role', 'fine_roles', 'count', alt.Tooltip('percentage:Q', format=".1f")]
            ).properties(
                width=300,
                height=300
            )

            # Faceted pie chart without legend
            pie_facet = base_no_legend.facet(
                column=alt.Column('main_role:N', title=None, header=alt.Header(labelAngle=0))
            ).resolve_scale(
                color='independent'
            ).properties(
                bounds='flush',
                title="Fine-Grained Roles per Main Role"
            )

            # Dummy chart with only the legend
            legend_chart = alt.Chart(role_fine_counts).mark_point().encode(
                y=alt.Y('fine_roles:N', axis=alt.Axis(title=None, labels=False, ticks=False)),
                color=alt.Color('fine_roles:N', scale=color_scale, legend=alt.Legend(title="Fine Role"))
            ).properties(width= 20, height=250)

            # Combine: pies + legend
            final_chart = alt.hconcat(pie_facet, legend_chart)

            st.altair_chart(final_chart, use_container_width=True)



    if check_pie_role:
        echarts_data = [
            {"value": int(row["count"]), "name": f"{row['main_role']} → {row['fine_roles']}"}
            for _, row in role_fine_counts.iterrows()
        ]

        # ECharts options
        options = {
            "title": {"text": "Fine-Grained Role Distribution", "left": "center"},
            "tooltip": {"trigger": "item"},
            "legend": {
                "orient": "vertical",
                "left": "left",
                "textStyle": {"color": "#333"}
            },
            "series": [
                {
                    "name": "Role Breakdown",
                    "type": "pie",
                    "radius": ["30%", "70%"],  # donut-style for clarity
                    "label": {
                        "show": True,
                        "formatter": "{b}\n({d}%)",
                        "fontSize": 12,
                        "color": "#000000",
                        "fontWeight": "bold",
                        "overflow": "truncate"
                    },
                    "labelLine": {"show": False},
                    "data": echarts_data,
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.5)"
                        }
                    },
                }
            ],
        }

        st_echarts(options=options, height="600px")





    # --- Network Graphs ---
    network_rows = []
    for i in range(column_count):
        selected_file = st.session_state.get(f"file_{i}")
        if selected_file and selected_file != "Select a file":
            labels = load_labels_stage2(selected_file, threshold)
            df_network = predict_entity_framing(labels, threshold)
            df_network = df_network[df_network['main_role'].isin(role_filter)]
            df_network = df_network.explode("fine_roles")
            df_network['fine_roles'] = df_network['fine_roles'].str.strip().str.title()
            df_network = df_network[df_network['fine_roles'].notnull() & (df_network['fine_roles'] != '')]
            df_network['article'] = selected_file
            network_rows.append(df_network)
    
    if not network_rows:
        st.warning("No data to display. Select file(s) with annotated content.")
        st.stop()
    
    else:
        st.header("Filtered Network Graph")

        graph_df = pd.concat(network_rows)
        graph_df['entity'] = graph_df['entity'].str.strip().str.title()
        graph_df['node_label'] = graph_df['entity'] + " (" + graph_df['fine_roles'] + ")"

        # -- Multiselect for restrict_entities --
        restrict_entities = st.multiselect(
            "Include entities",
            options=graph_df['entity'].unique(),
            default=graph_df['entity'].unique()
        )

        # Filter graph_df based on selected entities
        graph_df = graph_df[graph_df['entity'].isin(restrict_entities)]

        if graph_df.empty:
            st.info("No matching entities after filter.")
            st.stop()

        # Rebuild node labels post-filter
        graph_df["node_label"] = graph_df["entity"] + " (" + graph_df["fine_roles"] + ")"

        # -- Rebuild co-occurrence from filtered data --
        co_occurrence = Counter()
        for _, group in graph_df.groupby("article"):
            nodes = group["node_label"].unique()
            pairs = itertools.combinations(sorted(nodes), 2)
            co_occurrence.update(pairs)

        # -- Build the graph --
        G = nx.Graph()
        for node, row in graph_df.groupby("node_label").first().iterrows():
            G.add_node(node, role=row["main_role"])

        for (node1, node2), weight in co_occurrence.items():
            if node1 in G and node2 in G:
                G.add_edge(node1, node2, weight=weight)

        # -- Visualization --

        if G.number_of_nodes() == 0:
            st.info("No nodes to visualize.")
            st.stop()

        pos = nx.spring_layout(G, seed=42)
        max_weight = max(nx.get_edge_attributes(G, "weight").values(), default=1)

        edge_widths = []
        edge_colors = []
        for u, v in G.edges():
            w = G[u][v]["weight"]
            edge_widths.append(0.1 + (w**2.5))
            intensity = min(0.9, 0.3 + 0.7 * (w / max_weight))
            edge_colors.append(mcolors.to_hex((1 - intensity, 1 - intensity, 1 - intensity)))

        node_colors = [ROLE_COLORS.get(G.nodes[n]["role"], "#cccccc") for n in G.nodes()]

        if True:
            node_sizes = graph_df["node_label"].value_counts().to_dict()

            plt.figure(figsize=(14, 14))
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

            # Get x and y coordinates of all nodes
            x_vals, y_vals = zip(*pos.values())

            # Set axis limits with padding
            plt.xlim(min(x_vals) - 0.3, max(x_vals) + 0.3)
            plt.ylim(min(y_vals) - 0.3, max(y_vals) + 0.3)


            node_sizes = [node_sizes.get(node, 1) * 900 for node in G.nodes()]
            node_colors = [
                ROLE_COLORS.get(G.nodes[node]['role'], "#cccccc") for node in G.nodes()
            ]

            nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_widths)
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
            nx.draw_networkx_labels(G, pos, font_size=10)

            st.header("Network Graph Static")
            st.write("Explore the relationship between entities and roles across documents to see how groups of entity+role pairs can be used to identify potential narratives.")

            plt.axis("off")
            plt.tight_layout()
            st.pyplot(plt.gcf())


    # --- Interactive Network Graph ---
    if check_network_interactive and not graph_df.empty:
        st.header("Network Graph Interactive ")
        st.write("Interact with the connection between the entity,role nodes present across multiple documents. The edges represent a document where the two nodes are both present. ")

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
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        net.repulsion(node_distance=150, spring_length=200)
        net.show_buttons(filter_=["physics"])

        # Use temporary file properly
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            tmp_path = tmp_file.name

        with open(tmp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=1000, scrolling=True)

        # Optional cleanup
        os.remove(tmp_path)

st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")