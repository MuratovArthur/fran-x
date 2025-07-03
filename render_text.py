import streamlit as st
import pandas as pd
import re
from collections import defaultdict
import html as html_utils 
from collections import defaultdict
import html as html_utils
from sidebar import get_text_color
from rapidfuzz import fuzz, process

ROLE_COLORS = {
   "Protagonist": "#a1f4a1",
   "Antagonist":  "#f4a1a1",
   "Innocent":    "#a1c9f4",
}

def normalize_entities(graph_df, threshold=90):
    def clean_text(text):
        return text.strip().lower().replace(".", "").replace(",", "")

    # Manual alias handling first (cleaned)
    manual_aliases = {
        "us": "United States",
        "u.s.": "United States",
        "u.s": "United States",
        "usa": "United States",
        "united states of america": "United States",
        "the united states": "United States",
        "putin": "Vladimir Putin",
        "v putin": "Vladimir Putin",
        "west": "the west"
    }

    graph_df = graph_df.copy()  # Avoid modifying the original DataFrame
    graph_df['entity_clean'] = graph_df['entity'].apply(clean_text)

    # Start with manual mappings
    canonical_map = {k: v for k, v in manual_aliases.items()}

    # Use fuzzy matching to expand canonical map for unmapped entries
    entity_list = graph_df['entity_clean'].unique().tolist()
    for entity in entity_list:
        if entity in canonical_map:
            continue
        matches = process.extract(entity, list(entity_list), scorer=fuzz.token_sort_ratio)
        for match, score, _ in matches:
            if score >= threshold and match in canonical_map:
                canonical_map[entity] = canonical_map[match]
                break

    # Apply mapping
    graph_df['entity'] = graph_df['entity_clean'].map(canonical_map).fillna(graph_df['entity_clean'])
    graph_df['entity'] = graph_df['entity'].apply(lambda x: x.title())

    # Drop temp column now that we're done
    graph_df.drop(columns=['entity_clean'], inplace=True)

    return graph_df


def reformat_text_html_with_tooltips(text, labels_dict, hide_repeat=False, highlighted_word=None):
    spans = []
    entity_history = defaultdict(set)
    ##st.write(labels_dict)

    # Collect spans
    for entity, mentions in labels_dict.items():
        if not isinstance(mentions, list):
            st.stop()
            raise TypeError(f"Expected a list for entity '{entity}', but got {type(mentions)}")

        for mention in mentions:
            if not isinstance(mention, dict):
                st.stop()
                raise TypeError(f"Malformed mention at entity '{entity}': {mention}")

            ##st.write(mention)
            start = mention.get("start_offset", 7)
            end = mention.get("end_offset", 0)
            ##st.write(f"start:{start}, emd: {end}")

            start = max(0, min(start, len(text)))
            end = max(start, min(end, len(text)))

            main_role = mention.get("main_role", "")
            color = ROLE_COLORS.get(main_role, "#000000")
            fine_roles_set = frozenset(r.strip().title() for r in mention.get("fine_roles"))
            fine_roles_str = ", ".join(fine_roles_set)
            filtered_roles_str = ', '.join(
                role for role, score in mention.get('fine_roles').items()
            )

            is_repeated = fine_roles_set in entity_history[entity]
            adjusted_color = color
            if not (hide_repeat and is_repeated and color.startswith("#")):
                entity_history[entity].add(fine_roles_set)

            ##st.write(mention.get('confidence'))

            tooltip = (
                f"Role: {main_role or 'Unknown'}<br>"
                f"Confidence: {mention.get('confidence')}<br>"
                f"Fine roles: {mention.get('fine_roles')}"
            )

            entity_text = text[start:end].strip() or entity

            if highlighted_word:
                pattern = re.compile(re.escape(highlighted_word), flags=re.IGNORECASE)
                entity_text = pattern.sub(
                    r'<span style="border: 2px solid black; background-color: yellow; padding:1px 4px; margin: 0 1px; border-radius: 4px;">\g<0></span>',
                    entity_text
                )

            if hide_repeat and (hide_repeat == True and is_repeated):
                continue  # skip rendering this mention entirely
            else:
                ##st.write(f"start:{start}, emd: {end}")

                spans.append({
                    "start_offset": start,
                    "end_offset": end,
                    "html": (
                        f'<span class="entity" '
                        f'style="background-color:{adjusted_color}; padding:3px 6px; border-radius:4px;" '
                        f'data-tooltip="{tooltip}">'
                        f'{entity_text} | <span style="font-size: smaller;">{filtered_roles_str}</span></span>'
                    )
                })

    # Sort by start offset
    ##st.write(spans)
    spans.sort(key=lambda x: x["start_offset"])
    ##st.write(spans)
    # Assemble annotated text with optional highlighting
    result = []
    last_idx = 0
    for span in spans:
        start, end = span["start_offset"], span["end_offset"]
        ##st.write(f"start:{start}, emd: {end}")
        if start < last_idx:
            continue  # Overlap detected

        segment = text[last_idx:start]
        if highlighted_word:
            pattern = re.compile(re.escape(highlighted_word), flags=re.IGNORECASE)
            segment = pattern.sub(
                r'<span style="border: 2px solid black; background-color:yellow; padding:1px 4px; margin:0 1px; border-radius:4px;">\g<0></span>',
                segment
            )

        result.append(segment)
        result.append(span["html"])
        last_idx = end

    # Handle the tail segment
    tail = text[last_idx:]

    if highlighted_word:
        pattern = re.compile(re.escape(highlighted_word), flags=re.IGNORECASE)
        tail = pattern.sub(
            r'<span style="border: 2px solid black; background-color:yellow; padding:1px 4px; margin:0 1px; border-radius:4px;">\g<0></span>',
            tail
        )
    result.append(tail)

    #More strict version of searching        
    #if highlighted_word and False:
        #pattern = re.escape(highlighted_word)
        #tail = re.sub(
            #rf"(?<![\w])({pattern})(?=\b|'s|s\b|\W)"
            #r'<span style="border: 2px solid black; background-color:yellow; padding:1px 4px; margin:0 1px; border-radius:4px;">\1</span>',
            #tail,
            #flags=re.IGNORECASE
        #)
    #result.append(tail)

    annotated = ''.join(result)

    html = f"""
    <html><head>
    <script src="https://unpkg.com/@popperjs/core@2"></script>
    <script src="https://unpkg.com/tippy.js@6"></script>
    <link rel="stylesheet" href="https://unpkg.com/tippy.js@6/animations/scale.css"/>
    <style>
    body {{
        font-family: sans-serif;
        color: {get_text_color()};
    }}
    </style>
    </head><body>
    <div style="white-space: pre-wrap;">
    {annotated}
    </div>
    <script>
    tippy(".entity", {{
    content: reference => reference.getAttribute("data-tooltip"),
    allowHTML: true,
    trigger: "mouseenter click",
    interactive: true,
    animation: "scale"
    }});
    </script>
    </body></html>
    """

    return html



import pandas as pd
import streamlit as st

def predict_entity_framing(labels, threshold: float = 0.0):
    records = []

    for entity, mentions in labels.items():
        for mention in mentions:
            fine_roles = mention.get('fine_roles', {})
            if not fine_roles:
                continue  # Skip mentions with no fine roles

            filtered_roles = {
                role: confidence for role, confidence in fine_roles.items()
                if confidence >= threshold
            }

            if filtered_roles:
                records.append({
                    'entity': entity,
                    'main_role': mention.get('main_role', ''),
                    'fine_roles': filtered_roles,
                    'start_offset': mention.get('start_offset'),
                    'end_offset': mention.get('end_offset'),
                    'sentence': mention.get('sentence', '')
                })

    # Optional fallback if no records passed the threshold
    if not records:
        records.append({
            'entity': 'abcdef',
            'main_role': 'innocent',
            'fine_roles': {'forgotten': 0.0},
            'start_offset': 0,
            'end_offset': 0,
            'sentence': 'abcdef'
        })

    ##st.write(labels)
    ##st.write(records)
    return pd.DataFrame(records)


def row_per_role_entity_framing(labels, threshold: float = 0.0):
    records = []
    ##st.write(labels)

    for entity, mentions in labels.items():
        for mention in mentions:
            fine_roles = mention.get('fine_roles', {})
            if not fine_roles:
                continue  # Skip mentions with no fine roles

            for role, confidence in fine_roles.items():
                ##st.write(confidence)
                ##st.write(role)
                ##st.write(f'threshold of {threshold}')

                if confidence >= threshold:
                    ##st.write(f'passes threshold of {threshold}')
                    records.append({
                        'entity': entity,
                        'main_role': mention.get('main_role', ''),
                        'fine_roles': role,
                        'confidence': confidence,
                        'start': mention.get('start_offset'),
                        'end': mention.get('end_offset'),
                        'sentence': mention.get('sentence', '')
                    })

    # Optional fallback if no records passed the threshold
    if not records:
        records.append({
            'entity': 'abcdef',
            'main_role': 'innocent',
            'fine_roles': 'forgotten',
            'confidence': 0.0,
            'start': 0,
            'end': 0,
            'sentence': 'abcdef'
        })

    ##st.write(labels)
    ##st.write(records)
    return pd.DataFrame(records)



def format_sentence_with_spans(sentence_text, labels, threshold, hide_repeat=True, show_fine_roles=False, seen_fine_roles=None):
    if seen_fine_roles is None:
        seen_fine_roles = defaultdict(set)

    spans = []
    sentence_lower = sentence_text.lower()

    for entity, mentions in labels.items():
        entity_key = entity.strip().lower()

        for mention in mentions:
            if mention.get('confidence', 0) < threshold:
                continue
            if mention.get('sentence', '').strip() != sentence_text.strip():
                continue

            mention_text = entity.strip()
            if not mention_text:
                continue

            match_start = sentence_lower.find(mention_text.lower())
            if match_start == -1:
                continue
            match_end = match_start + len(mention_text)

            main_role = mention.get('main_role', '')
            base_color = ROLE_COLORS.get(main_role, "#000000")

            fine_roles_raw = mention.get('fine_roles', [])
            fine_roles_set = frozenset(r.strip().title() for r in fine_roles_raw)
            fine_roles_str = ", ".join(fine_roles_set)

            is_repeated = fine_roles_set in seen_fine_roles[entity_key]
            seen_fine_roles[entity_key].add(fine_roles_set)

            if hide_repeat and is_repeated and base_color.startswith("#") and len(base_color) == 7:
                r = int(base_color[1:3], 16)
                g = int(base_color[3:5], 16)
                b = int(base_color[5:7], 16)
                background_color = f"rgba({r}, {g}, {b}, 0.3)"
            else:
                background_color = base_color

            mention_display = html_utils.escape(sentence_text[match_start:match_end])

            if show_fine_roles:
                fine_roles_html = f' | <span style="font-size:smaller; opacity:0.75;">{fine_roles_str}</span>'
            else:
                fine_roles_html = ""

            span_html = (
                f'<span style="background-color:{background_color}; padding:3px 6px; border-radius:4px;">'
                f'{mention_display}{fine_roles_html}</span>'
            )

            spans.append({
                "start": match_start,
                "end": match_end,
                "html": span_html
            })

    # Merge spans
    spans.sort(key=lambda x: x["start"])
    result = []
    last_idx = 0
    for span in spans:
        start = span["start"]
        end = span["end"]
        if start < last_idx:
            continue
        result.append(html_utils.escape(sentence_text[last_idx:start]))
        result.append(span["html"])
        last_idx = end
    result.append(html_utils.escape(sentence_text[last_idx:]))

    body = ''.join(result)

    full_html = (
        '<div class="franx-block">'
        f'<style>.franx-block {{ font-family: sans-serif; color: {get_text_color()}; }}</style>'
        f'<div style="white-space: pre-wrap;">{body}<br></div>'
        '</div>'
    )


    return full_html, seen_fine_roles