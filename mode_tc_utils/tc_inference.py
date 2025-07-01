import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline



# Constants
MODEL_PATH = "artur-muratov/franx-cls"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
PIPELINE = pipeline("text-classification", model=MODEL, tokenizer=TOKENIZER,  return_all_scores=True)



def run_role_inference(df, clf_pipeline=PIPELINE, threshold=0.01, margin=0.05):
    """
    Adds predicted role scores and filtered roles (within margin of top score) to the dataframe.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'entity_mention', 'p_main_role', and 'context' columns.
        clf_pipeline (transformers.Pipeline): HuggingFace classification pipeline.
        threshold (float): Minimum confidence to retain a role score.
        margin (float): Margin within top confidence to select final roles.
    
    Returns:
        pd.DataFrame: Updated with 'predicted_roles' and 'predicted_fine_margin' columns.
    """

    def pipeline_with_confidence(example, threshold=threshold):
        input_text = (
            f"Entity: {example['entity_mention']}\n"
            f"Main Role: {example['p_main_role']}\n"
            f"Context: {example['context']}"
        )
        try:
            scores = clf_pipeline(input_text)[0]  # [{'label': ..., 'score': ...}]
        except Exception as e:
            print(f"Error in pipeline: {e}")
            return {}

        filtered_scores = {
            s['label']: round(s['score'], 4) for s in scores if s['score'] > threshold
        }
        return dict(sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True))

    def select_roles_within_margin(role_scores, margin=margin):
        if not isinstance(role_scores, dict) or not role_scores:
            return []
        top_score = max(role_scores.values())
        return [
            role for role, score in role_scores.items()
            if abs(score - top_score) <= margin
        ]
    
    def get_roles_with_scores(role_scores, selected_roles):
        return {role: role_scores[role] for role in selected_roles if role in role_scores}

    df['predicted_roles'] = df.apply(lambda row: pipeline_with_confidence(row.to_dict()), axis=1)
    df['predicted_fine_margin'] = df['predicted_roles'].apply(select_roles_within_margin)
    df['predicted_fine_with_scores'] = df.apply(
        lambda row: get_roles_with_scores(row['predicted_roles'], row['predicted_fine_margin']),
        axis=1
    )

    return df



