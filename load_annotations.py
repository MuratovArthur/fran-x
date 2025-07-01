import csv
from collections import defaultdict
from pathlib import Path
import streamlit as st

def capitalize(s):
    return s.capitalize() if s else s

def load_article(filepath):
    if filepath.endswith("Select a file"):
        return ""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_labels_old(folder_name, article_file_name, threshold):
    role_timeline = defaultdict(list)
    csv_files = [str(f) for f in Path(folder_name).iterdir() if f.is_file()]

    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['article_id'] != article_file_name:
                    continue

                try:
                    confidence = float(row.get('confidence', 0.85))
                    if confidence < threshold:
                        continue

                    entity = row['entity_mention'].strip()
                    main_role = capitalize(row['main_role'].strip())

                    raw_roles = row['fine_grained_roles'].strip()
                    fine_roles = [r.strip() for r in raw_roles[2:-2].replace("'", "").split(",") if r.strip()]

                    sentence = row.get('context_sentence', '').strip()

                    start_offset = int(row['start_offset'])
                    end_offset = int(row['end_offset'])

                    role_entry = {
                        'start_offset': start_offset,
                        'end_offset': end_offset,
                        'main_role': main_role,
                        'fine_roles': sorted(fine_roles),
                        'confidence': confidence,
                        'sentence': sentence
                    }

                    role_timeline[entity].append(role_entry)

                except (ValueError, KeyError):
                    continue  # Skip malformed rows

    # Sort mentions per entity by offset
    for mentions in role_timeline.values():
        mentions.sort(key=lambda x: x['start_offset'])

    return dict(role_timeline)




import csv
from pathlib import Path
from collections import defaultdict
from ast import literal_eval
import re

def capitalize(text):
    return text.capitalize() if isinstance(text, str) else text

def safe_role_list(value):
    try:
        parsed = literal_eval(value)
        if isinstance(parsed, (set, list)):
            return [str(r).strip() for r in parsed if str(r).strip()]
    except:
        pass
    return re.findall(r"'([^']+)'", value)

import ast

def safe_margin_dict(raw_margin_str):
    try:
        return ast.literal_eval(raw_margin_str)
    except (ValueError, SyntaxError):
        return {}


def safe_fine_roles_dict(value):
    try:
        parsed = literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except:
        return {}
    
#"{'Deceiver': 0.3147, 'Corrupt': 0.2053, 'Incompetent': 0.1512, 'Conspirator': 0.1025, 'Bigot': 0.0655}"
def load_labels(folder_name, article_file_name, threshold=0.0):
    role_timeline = defaultdict(list)
    csv_files = [str(f) for f in Path(folder_name).iterdir() if f.is_file()]

    n = 0
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['article_id'] != article_file_name:
                    continue

                try:
                    entity = row['entity_mention'].strip()
                    main_role = capitalize(row['main_role'].strip())

                    #st.write(safe_fine_roles_dict(row['predicted_roles']))

                    #st.write(sorted(safe_fine_roles_dict(row['predicted_roles']).items()))


                    sentence = row.get('context', '').strip()
                    start_offset = int(row['start_offset'])
                    adjusted_end = int(row['adjusted_end'])

                    #st.write(0)
                    #st.write(sorted(safe_fine_roles_dict(row['predicted_roles']).items()))





                    fine_roles_dict = safe_fine_roles_dict(row['predicted_roles'])
                    sorted_roles = sorted(fine_roles_dict.items(), key=lambda x: x[1], reverse=True) 
                    #FOR NOW JUST TOP 1 CHANGE ONCE THE COLUMN IS CUT DOWN





                    if sorted_roles:
                        top_role, top_score = sorted_roles[0]
                        if top_score >= threshold:
                            role_entry = {
                                'start_offset': start_offset,
                                'end_offset': adjusted_end,
                                'main_role': main_role,
                                'fine_roles': (top_role, top_score),
                                'sentence': sentence
                            }

                            role_timeline[entity].append(role_entry)



                except (ValueError, KeyError, SyntaxError):
                    continue

    # Sort entries by start offset
    for mentions in role_timeline.values():
        mentions.sort(key=lambda x: x['start_offset'])

    #st.write(role_timeline)

    return role_timeline
