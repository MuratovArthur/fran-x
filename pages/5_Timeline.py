import streamlit as st
from sidebar import render_sidebar, ROLE_COLORS
from render_text import predict_entity_framing, normalize_entities
import json

st.set_page_config(page_title="FRaN-X", layout="wide")
st.title("In-Depth Timeline")
st.write("See how each entity changes its main role and fine grain role over time")

article, labels, user_folder, threshold, role_filter, hide_repeat = render_sidebar()

def highlight_fine_roles(sentence, roles, color):
   if not isinstance(roles, list):
       roles = [roles]
   for role in roles:
       if role and isinstance(role, str) and role in sentence:
           sentence = sentence.replace(
               role,
               f"<span style='background-color:{color}; padding:2px 6px; border-radius:4px; font-weight:bold;'>{role}</span>"
           )
   return sentence


def render_block(block, role_main, role_fine, count, color):
    # role_fine is expected to be a list of dicts like [{ "Terrorist": 0.5408 }]
    #st.write(role_fine)

    # Extract just the role names for display
    if isinstance(role_fine, list):
        fine_display = ", ".join([list(r.keys())[0] for r in role_fine if isinstance(r, dict)])
    else:
        fine_display = str(role_fine)

    with st.expander(f"{role_main} / {fine_display} ({count} instances)", expanded=True):
        for b in block:
            fine_roles = b['fine_roles']  # Expects a dict: {'Role': confidence}
            
            ##st.write(f"fine_roles: {fine_roles}")

            # If fine_roles is a dict, extract the role and confidence
            if isinstance(fine_roles, dict):
                role, confidence = next(iter(fine_roles.items()))
            else:
                role, confidence = str(fine_roles), 0.0

            highlighted_sentence = highlight_fine_roles(b['sentence'], [role], color)

            ##st.write(highlighted_sentence)

            highlighted_role_html = (
                f"<span style='background-color:{color}; padding:2px 6px; border-radius:4px; font-weight:bold;'>{role}</span>"
            )

            ##st.write(highlighted_role_html, unsafe_allow_html=True)

            st.markdown(
                f"""<div style='padding:10px; border-radius:5px; margin-bottom:5px; border: 1px solid #ddd;'>
                <b>Fine Role(s):</b> {highlighted_role_html}<br>
                <b>Confidence:</b> {confidence:.2f}<br>
                <b>Sentence:</b> {highlighted_sentence}</div>""",
                unsafe_allow_html=True
            )




if article and labels:
    df_f = predict_entity_framing(labels, threshold)
    ##st.write(df_f)
    df_f = df_f[df_f['main_role'].isin(role_filter)]

    #df_f = normalize_entities(df_f, 10)

    df_f.sort_values(by=["entity", "start_offset"], inplace=True)

    ##st.write(df_f)
    entity_order = df_f.groupby("entity")["start_offset"].min().sort_values().index.tolist()

    ##st.write(entity_order)

    for entity in entity_order:
        group = df_f[df_f["entity"] == entity]
        total_annotations = len(group)
        st.divider()
        st.markdown(f"#### {entity} (total: {total_annotations})")

        block = []
        prev_main = prev_fine = None
        count = 0

        for _, row in group.iterrows():
            main = row["main_role"]
            fine = row["fine_roles"] if isinstance(row["fine_roles"], list) else [row["fine_roles"]]

            if (main, fine) != (prev_main, prev_fine) and prev_main is not None:
                # Insert role change indicator

                color = ROLE_COLORS.get(prev_main, "#ccc")
                render_block(block, prev_main, prev_fine, count, color)

                prev_fine_display = ", ".join([list(f.keys())[0] for f in prev_fine if isinstance(f, dict)])
                new_fine_display = ", ".join([list(f.keys())[0] for f in fine if isinstance(f, dict)])

                st.markdown(f"""
                <div style='background-color:rgba(249, 249, 249, 0.7); border-left:4px solid {ROLE_COLORS.get(main)}; padding:10px; margin:10px 0; border-radius:6px; font-size:15px;'>
                <b>↪️ Role Change {entity}</b><br>
                <b>From:</b> {prev_main} / {prev_fine_display}<br>
                <b>To:</b> {main} / {new_fine_display}
                </div>
                """, unsafe_allow_html=True)


                block = []
                count = 0

            block.append(row)
            count += 1
            prev_main, prev_fine = main, fine

        if block:
            color = ROLE_COLORS.get(prev_main, "#ccc")

            render_block(block, prev_main, prev_fine, count, color)

st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")