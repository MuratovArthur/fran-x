import os
import pandas as pd
import pandas as pd
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize


def char_window_context(text, start_offset, end_offset, window=150):
    try:
        start_offset = int(start_offset)
        end_offset = int(end_offset)
    except (ValueError, TypeError):
        return text[:2 * window].strip()

    left = max(0, start_offset - window)
    right = min(len(text),end_offset + window)
    return text[left:right].strip()




def convert_prediction_txt_to_csv(article, prediction_file, article_text, output_csv):
    """
    Converts a Stage 1 prediction .txt into a CSV with full article text.
    """
    records = []
    with open(prediction_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            #article_id = parts[0]
            entity_mention = parts[0]
            start_offset = parts[1]
            end_offset = parts[2]
            p_main_role = parts[3]
            entity_mention, start_offset, end_offset, p_main_role = parts


            context = char_window_context(article, start_offset, end_offset)
            #sentence = sentence_window_context(article, start_offset, end_offset)
            records.append({
                #"article_id": article_id,
                "text": article,
                "entity_mention": entity_mention,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "p_main_role": p_main_role,
                "context": context
            })
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8")
