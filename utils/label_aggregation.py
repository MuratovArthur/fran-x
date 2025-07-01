import warnings
from typing import Optional, Tuple

import numpy as np


class AggregationStrategy:
    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"


class PredictionAggregator:
    def __init__(self, classifier):
        self.model = classifier.model
        self.tokenizer = classifier.tokenizer

    def gather_pre_entities(
            self,
            sentence: str,
            input_ids: np.ndarray,
            scores: np.ndarray,
            offset_mapping: Optional[list[Tuple[int, int]]],
            special_tokens_mask: np.ndarray,
            aggregation_strategy: str,
    ) -> list[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens, they should only occur
            # at the sentence boundaries since we're not encoding pairs of
            # sentences so we don't have to keep track of those.
            if special_tokens_mask[idx]:
                continue

            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                word_ref = sentence[start_ind:end_ind]
                if 'layoutlm' in getattr(self.model, 'base_model_prefix', ''):
                    is_subword = start_ind != 0
                elif getattr(
                        self.tokenizer._tokenizer.model, "continuing_subword_prefix", None
                ):
                    # This is a BPE, word aware tokenizer, there is a correct way
                    # to fuse tokens
                    is_subword = len(word) != len(word_ref)
                else:
                    # This is a fallback heuristic. This will fail most likely on any kind of text + punctuation mixtures that will be considered "words". Non word aware models cannot do better than this unfortunately.
                    if aggregation_strategy in {
                        AggregationStrategy.FIRST,
                        AggregationStrategy.AVERAGE,
                        AggregationStrategy.MAX,
                    }:
                        warnings.warn(
                            "Tokenizer does not support real words, using fallback heuristic",
                            UserWarning,
                        )
                    is_subword = (
                        sentence[start_ind - 1: start_ind] != " "
                        if start_ind > 0
                        else False
                    )

                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                start_ind = None
                end_ind = None
                is_subword = False

            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities

    def aggregate_word(
            self, entities: list[dict], aggregation_strategy: AggregationStrategy
    ) -> dict:
        word = self.tokenizer.convert_tokens_to_string(
            [entity["word"] for entity in entities]
        )
        if aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.model.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity

    def aggregate_words(
            self, entities: list[dict], aggregation_strategy: AggregationStrategy
    ) -> list[dict]:
        """
        Override tokens from a given word that disagree to force agreement on word boundaries.

        Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft|
        company| B-ENT I-ENT
        """
        assert aggregation_strategy not in {
            AggregationStrategy.NONE,
            AggregationStrategy.SIMPLE,
        }, "NONE and SIMPLE strategies are invalid"

        word_entities = []
        word_group = None
        for entity in entities:
            if word_group is None:
                word_group = [entity]
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(
                    self.aggregate_word(word_group, aggregation_strategy)
                )
                word_group = [entity]
        # Last item
        word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
        return word_entities

    def aggregate(
            self, pre_entities: list[dict], aggregation_strategy: AggregationStrategy
    ) -> list[dict]:
        if aggregation_strategy in {
            AggregationStrategy.NONE,
            AggregationStrategy.SIMPLE,
        }:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.model.config.id2label[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            entities = self.aggregate_words(pre_entities, aggregation_strategy)

        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities(entities)

    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # It's not in B-, I- format
            # Default to I- for continuation.
            bi = "I"
            tag = entity_name
        return bi, tag

    def group_sub_entities(self, entities: list[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def group_entities(self, entities: list[dict]) -> list[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        for entity in entities:
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            # If the current entity is similar and adjacent to the previous entity,
            # append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" prefixes
            # Shouldn't merge if both entities are B-type
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])

            if tag == last_tag and bi != "B":
                # Modify subword type to be previous_type
                entity_group_disagg.append(entity)
            else:
                # If the current entity is different from the previous entity
                # aggregate the disaggregated entity group
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            # it's the last entity, add it to the entity groups
            entity_groups.append(self.group_sub_entities(entity_group_disagg))

        return entity_groups

    def aggregate_predictions(
            self,
            input_ids,
            logits,
            offset_mapping,
            special_tokens_mask,
            aggregation_strategy="simple",
    ):
        """
        The method performs aggregation of labels from token level to entity level for a batch of data.
        Different aggregation strategies may be used here as hyperparameters to validate.
        :param input_ids: np.array - produced by the tokenizer of your model by default
        :param logits: np.array
        :param offset_mapping: np.array - produced by the tokenizer of your model if parameter return_offset_mapping == True
        :param special_tokens_mask: np.array - produced by the tokenizer of your model if parameter return_special_tokens_mask == True
        :param aggregation_strategy: Set['none', 'simple', 'first', 'max', 'average']
        :return:
        """

        batch_results = []
        for idx in range(input_ids.shape[0]):
            sentence = self.tokenizer.decode(input_ids[idx], skip_special_tokens=True)
            pre_entities = self.gather_pre_entities(
                sentence=sentence,
                input_ids=input_ids[idx],
                scores=logits[idx],
                offset_mapping=offset_mapping[idx],
                special_tokens_mask=special_tokens_mask[idx],
                aggregation_strategy=aggregation_strategy,
            )
            aggregated_results = self.aggregate(
                pre_entities=pre_entities, aggregation_strategy=aggregation_strategy
            )
            batch_results.append(aggregated_results)
        return batch_results
