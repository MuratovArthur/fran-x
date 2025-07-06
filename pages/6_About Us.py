import streamlit as st


role_descriptions = {
    "Protagonist": {
        "Guardian": "Heroes or guardians who protect values or communities, ensuring safety and upholding justice.",
        "Martyr": "Individuals who sacrifice their well-being, or even their lives, for a greater good or cause.",
        "Peacemaker": "Individuals who advocate for harmony, resolving conflicts and bringing about peace.",
        "Rebel": "Revolutionaries who challenge the status quo and fight for significant change or liberation.",
        "Underdog": "Entities who, despite a disadvantaged position, strive against greater forces and obstacles.",
        "Virtuous": "Individuals portrayed as righteous, fair, and upholding high moral standards."
    },
    "Antagonist": {
        "Instigator": "Those who initiate conflict and provoke violence or unrest.",
        "Conspirator": "Individuals involved in plots and covert activities to undermine or deceive others.",
        "Tyrant": "Leaders who abuse their power, ruling unjustly and oppressing others.",
        "Foreign Adversary": "Entities from other nations creating geopolitical tension and acting against national interests.",
        "Traitor": "Individuals who betray a cause or country, seen as disloyal and treacherous.",
        "Spy": "Individuals engaged in espionage, gathering and transmitting sensitive information.",
        "Saboteur": "Those who deliberately damage or obstruct systems to cause disruption.",
        "Corrupt": "Individuals or entities engaging in unethical or illegal activities for personal gain.",
        "Incompetent": "Entities causing harm through ignorance, lack of skill, or poor judgment.",
        "Terrorist": "Individuals who engage in violence and terror to further ideological ends.",
        "Deceiver": "Manipulators who twist the truth, spread misinformation, and undermine trust.",
        "Bigot": "Individuals accused of hostility or discrimination against specific groups."
    },
    "Innocent": {
        "Forgotten": "Marginalized groups who are overlooked and ignored by society.",
        "Exploited": "Individuals or groups used for others’ gain, often without consent.",
        "Victim": "People suffering harm due to circumstances beyond their control.",
        "Scapegoat": "Entities unjustly blamed for problems or failures to divert attention."
    }
}


st.set_page_config(page_title="FRaN-X", initial_sidebar_state='expanded', layout="wide")
st.title("About Us")

st.markdown("### Gain a thorough understanding of news articles through the lens of entity framing.")

st.write("Most automated media‑analysis pipelines stop at detecting who is mentioned and whether coverage is positive or negative. They rarely ask how the news positions those actors (e.g., protagonist, antagonist, innocent). This project fills that gap by building an end‑to‑end system that analyzes role framing and presents them through an interactive web interface.")


st.markdown("#### Pages")

st.markdown("""

- **Home:** Input an article by pasting text or providing a URL. Specify a filename and process the article to generate structured lists of predicted entities and their roles.
- **Analysis:** View annotated articles with entities color-coded by their main roles and fine-grained roles displayed alongside. Hovering over entities reveals metadata, including confidence scores. Users can toggle repeated annotations, adjust a confidence threshold slider, and extract all sentences for a specific label to compare entity characterization across contexts.
- **Dynamic Analysis:** Load and compare up to four articles side by side to examine how different media outlets frame the same event or entity. This enables users to understand framing variations across sources.
- **Aggregate Analysis:** Explore aggregate-level insights through a network graph connecting entities and their roles across multiple documents. Users can filter nodes and edges by entity or document, and interactively manipulate the graph for deeper analysis.
- **Search:** Select one or more processed articles and search for specific words or phrases. All matches are highlighted within the article context, enabling quick localization of entities or terms of interest.
- **Timeline:** Visualize the evolution of entity roles over time. For each selected entity, view the sequence of fine-grained roles, confidence scores, and context sentences. The system highlights transitions in role assignments for multiple entities.

""")

st.markdown("#### Annotations and Taxonomy")

url = "https://arxiv.org/abs/2502.14718"
st.write("FRaN-X's annotation style is derived from the unique taxonomy used in the paper [_Entity Framing and Role Portrayal in the News_](%s) which classifies entities under 3 main roles: Protagonist, Antagonist, and Innocent, and 22 fine-grain roles." % url)

# Display using expanders
for main_role, sub_roles in role_descriptions.items():
    with st.expander(main_role.upper()):
        for sub_role, description in sub_roles.items():
            st.markdown(f"**• {sub_role}**: {description}")



st.divider()
st.markdown("*UGRIP 2025 FRaN-X Team* ")