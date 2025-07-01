import csv
from collections import defaultdict
from pathlib import Path


def capitalize(s):
    return s.capitalize() if s else s

def load_article(filepath):
    if filepath.endswith("Select a file"):
        return ""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_labels(folder_name, article_file_name, threshold):
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
