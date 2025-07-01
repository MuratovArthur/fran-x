# utils/bert.py

from typing import Optional, Union, List

from datasets import Dataset
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.models.layoutlmv3.tokenization_layoutlmv3_fast import \
    LayoutLMv3TokenizerFast

from src.schema import BERTDataset


def align_labels_with_tokens(labels: list[int], word_ids: list[Optional[int]]):
    """
    Aligns the given NER (Named Entity Recognition) labels with tokenized word_ids.

    Args:
        labels (list[int]): list of NER labels.
        word_ids (list[Optional[int]]): list of word_ids obtained after tokenization.

    Returns:
        list[int]: list of aligned labels.
    """
    # TODO: Implement the logic for alignment labels with corresponding tokens
    # You can find reference implementation in the manual here : https://huggingface.co/learn/nlp-course/chapter7/2#processing-the-data
    # YOUR CODE HERE #
    
    new_labels = []
    current_word = None
    
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word:
            new_labels.append(labels[word_id])
            current_word = word_id
        else:
            if labels[word_id] % 2 == 1:
                labels[word_id] += 1
            new_labels.append(labels[word_id])
    
    return new_labels

def _align_word_labels_to_tokens(
    word_labels: List[int],
    word_ids: List[int],
) -> List[int]:
    """
    • keep the label on the **first** sub-token of a word
    • mask all subsequent sub-tokens with -100 so they don't contribute
      to the loss.
    """
    label_ids = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:                     # special token
            label_ids.append(-100)
        elif word_idx != previous_word_idx:      # first sub-token ⇒ keep original label
            label_ids.append(word_labels[word_idx])
        else:
            # subsequent sub-token: use same entity but switch B->I if needed
            lbl = word_labels[word_idx]
            if lbl == 0:                         # O stays O
                label_ids.append(0)
            elif lbl % 2 == 1:                   # B-xxx → I-xxx (odd to even)
                label_ids.append(lbl + 1)
            else:                                # already I-xxx
                label_ids.append(lbl)
        previous_word_idx = word_idx

    return label_ids


def tokenize_and_align_labels(
    examples: "BERTDataset",
    tokenizer: Union[BertTokenizerFast, LayoutLMv3TokenizerFast],
    max_length: int = 512,
    doc_stride: int = 128,
) -> Dataset:
    """
    Tokenise *and* break long documents into overlapping windows,
    then align BIO labels to every token.

    Parameters
    ----------
    examples : mapping with keys ``"tokens"`` and ``"ner_tags"``
    tokenizer : HF fast tokenizer
    max_length : int
        Maximum sequence length for each window (default 512).
    doc_stride : int
        Overlap between successive windows (default 128).

    Returns
    -------
    datasets.Dataset
        Encoded input_ids / attention_mask / labels (token-level).
    """
    # -- 1. Tokenise with overflow so we cover the whole doc ------------
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding=False,
        truncation=True,
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=False,   # we don't need char-spans here
    )

    # -- 2. Map each resulting window back to its *original* example ----
    sample_map = tokenized.pop("overflow_to_sample_mapping")

    # -- 3. Align BIO labels window-by-window ---------------------------
    new_labels = []
    for i in range(len(tokenized["input_ids"])):
        orig_idx       = sample_map[i]                 # which doc
        word_ids       = tokenized.word_ids(batch_index=i)
        word_lvl_labels = examples["ner_tags"][orig_idx]

        label_ids = _align_word_labels_to_tokens(word_lvl_labels, word_ids)
        new_labels.append(label_ids)

    tokenized["labels"] = new_labels

    # -- 4. Return as a Dataset --------------------------------------
    return Dataset.from_dict(tokenized)