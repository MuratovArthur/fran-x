from typing import TypedDict, List


class BERTDataset(TypedDict):
    """Dataset format for BERT model."""
    tokens: List[List[str]]
    ner_tags: List[List[int]]