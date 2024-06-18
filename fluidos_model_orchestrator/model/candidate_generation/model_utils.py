import ast
import random
import sys
from typing import Any

import torch  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from torch import Tensor


def intent_to_value(intent_name: str, intent_value: Any) -> Any:
    translated_value: Any = intent_value
    if intent_value != "empty" and intent_value != "none":
        if intent_name == "fluidos-intent-throughput":
            translated_value = float(intent_value[:-2]) if isinstance(intent_value, str) else intent_value
        elif intent_name == "cpu":
            translated_value = int(intent_value[:-1]) if isinstance(intent_value, str) else intent_value
        elif intent_name == "memory":
            translated_value = int(intent_value[:-2]) if isinstance(intent_value, str) else intent_value
        elif intent_name == "fluidon-intent-location":
            translated_value = intent_value
    return translated_value


def compute_embedding_for_sentence(sentence: str, sentence_transformer: SentenceTransformer) -> Tensor:
    embeddings = sentence_transformer.encode(sentence)
    return torch.tensor(embeddings).unsqueeze(0)


def intents_dict_to_values(input_intents: dict[str, Any]) -> dict[str, Any]:
    intent_to_values: dict[str, Any] = {}
    for intent in input_intents:
        intent_to_values[intent] = intent_to_value(intent, input_intents[intent])

    return intent_to_values


def check_config_for_relevance(intent_to_values: dict[str, Any], feature: dict[str, Any]) -> tuple[bool, int]:
    relevant = True
    feature_props_values_sum = 0
    for intent_name, intent_value in intent_to_values.items():
        if not isinstance(intent_value, str):
            feature_props_values_sum += feature[intent_name]
            if intent_value > feature[intent_name]:
                relevant = False
                break
    return relevant, feature_props_values_sum


def find_matching_configs(input_intents: dict[str, Any],
                          configuration2id: dict[str, int], negative_location: str = 'd') -> tuple[list[int], list[int], int]:
    random.seed(42)
    relevant_configs_full: list[int] = []
    non_relevant_configs_full: list[int] = []
    minumal_value_config = -1
    min_resources = sys.maxsize
    intent_to_values = intents_dict_to_values(input_intents)

    if 'fluidos-intent-location' in intent_to_values:
        for feature_str in configuration2id:
            if feature_str != "none":
                feature = intents_dict_to_values(ast.literal_eval(feature_str))
                relevant = True
                if feature['fluidos-intent-location'] == intent_to_values['fluidos-intent-location']:
                    if negative_location == intent_to_values['fluidos-intent-location']:
                        relevant_configs_full.append(configuration2id[feature_str])
                    else:
                        relevant, feature_props_values_sum = check_config_for_relevance(intent_to_values, feature)
                        if relevant:
                            relevant_configs_full.append(configuration2id[feature_str])
                            if feature_props_values_sum < min_resources:
                                min_resources = feature_props_values_sum
                                minumal_value_config = configuration2id[feature_str]
                else:
                    relevant = False
                if not relevant:
                    non_relevant_configs_full.append(configuration2id[feature_str])
            else:
                non_relevant_configs_full.append(configuration2id[feature_str])
    else:
        for feature_str in configuration2id:
            if feature_str != "none":
                feature = intents_dict_to_values(ast.literal_eval(feature_str))
                relevant, feature_props_values_sum = check_config_for_relevance(intent_to_values, feature)
                if relevant:
                    relevant_configs_full.append(configuration2id[feature_str])
                    if feature_props_values_sum < min_resources:
                        min_resources = feature_props_values_sum
                        minumal_value_config = configuration2id[feature_str]
                if not relevant:
                    non_relevant_configs_full.append(configuration2id[feature_str])

    return relevant_configs_full, non_relevant_configs_full, minumal_value_config
