import ast
from typing import Any

from fluidos_model_orchestrator.model.utils import FLUIDOS_COL_NAMES
from fluidos_model_orchestrator.model.utils import FLUIDOS_INPUT_OUTPUT_NAME


class FEEDBACK_STATUS:
    OK = "OK"
    FAIL = "FAIL"


def resource_dict_to_values(resources: dict[str, Any]) -> dict[str, Any]:
    resource_to_values: dict[str, Any] = {}
    for resource in resources:
        resource_to_values[resource] = resource_to_value(resource, resources[resource])

    return resource_to_values


def resource_to_value(resource_name: str, resource_value: Any) -> Any:

    translated_value: Any = resource_value
    if resource_value != "empty" and resource_value != "none":
        match resource_name:
            case FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_THROUGHPUT:
                translated_value = float(resource_value[:-2]) if isinstance(resource_value, str) else resource_value
            case FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU:
                translated_value = int(resource_value[:-1]) if isinstance(resource_value, str) else resource_value
            case FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY:
                translated_value = int(resource_value[:-2]) if isinstance(resource_value, str) else resource_value
            case FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_LOCATION:
                translated_value = resource_value
    return translated_value


def check_config_for_relevance(intent_to_values: dict[str, Any], feature: dict[str, Any]) -> tuple[bool, float]:
    relevant = True
    feature_props_values_sum = 0.0
    # NOTE:  it is expected that resource always contains cpu and memory indormation
    for intent_name, intent_value in intent_to_values.items():
        if intent_value != '-1' and intent_value != -1:
            resource_name = intent_name.replace(FLUIDOS_INPUT_OUTPUT_NAME.POD_NAME, FLUIDOS_INPUT_OUTPUT_NAME.TEMPLATE_RESOURCE_NAME)
            if resource_name in feature:
                if feature[resource_name]:
                    if not isinstance(intent_value, str):
                        feature_props_values_sum += feature[resource_name]
                        if intent_value > feature[resource_name]:
                            relevant = False
                            break
                    else:
                        if intent_value != feature[resource_name]:
                            relevant = False
                            break
    return relevant, feature_props_values_sum


def tr2id_from_str_to_list(template_resources2id: dict[str, int]) -> list[str | dict[str, Any]]:

    max_id = sorted(template_resources2id.values())[-1]
    config_list: list[str | dict[str, Any]] = [{} for i in range(max_id + 1)]
    for key, val in template_resources2id.items():
        feature_str = key
        if feature_str != "none":
            feature = resource_dict_to_values(ast.literal_eval(feature_str))  # type: ignore
            feature = ast.literal_eval(feature_str)
            config_list[val] = feature  # type: ignore
        else:
            config_list[val] = feature_str  # type: ignore
    return config_list
