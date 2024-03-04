from typing import Any

from ..common import ModelInterface
from ..common import ModelPredictRequest
from ..common import Intent
from ..common import KnownIntent
from .dummy import DummyOrchestrator
from .candidate_generation import Orchestrator as CGOrchestrator
from kopf import PermanentError

import logging


logger = logging.getLogger(__name__)


_model_instances: dict[str, ModelInterface] = {
    "dummy": DummyOrchestrator(),
    "candidate_generation": CGOrchestrator(),
}


def get_model_object(request: ModelPredictRequest) -> ModelInterface:
    logger.info(f"Retrieving model interface for {request}")

    return _model_instances["dummy"]


def convert_to_model_request(spec: Any) -> ModelPredictRequest:
    logger.info("Converting incoming custom resource to model request")

    if spec["kind"] == "Deployment":
        logger.debug("Processing Deployment object")
        return ModelPredictRequest(
            id=f"{spec['metadata']['name']}",
            pod_request=spec["spec"]["template"],
            intents=_extract_intents(spec["metadata"].get("annotations", {})),
        )

    if spec["kind"] == "Pod":
        logger.debug("Processing Pod object")

        # pod_dict["spec"]["containers"][0]["resources"]["requests"]
        intents = _extract_intents(spec["metadata"].get("annotations", {}))
        intents.extend(
            _extract_resource_intents(
                spec["spec"]["containers"][0]["resources"]["requests"]
            )
        )

        return ModelPredictRequest(
            id=spec["metadata"]["name"],
            pod_request=spec,
            intents=intents,
        )

    logger.error(f"Unsupported kind {spec['kind']}")
    raise PermanentError(f"Unsupported kind {spec['kind']}")


def _extract_resource_intents(requests: dict[str, str]) -> list[Intent]:
    logger.debug("Extracting resources intents from request")
    # pod_dict["spec"]["containers"][0]["resources"]["requests"]
    return [Intent("cpu", requests["cpu"]), Intent("memory", requests["memory"])]


def _extract_intents(annotations: dict[str, str]) -> list[Intent]:
    logger.debug("Extracting explicit intens from annotations")
    intents = []

    for key, value in annotations.items():
        if key.casefold().startswith("fluidos-intent-"):
            intent_name = _extract_intent_name(key)
            if KnownIntent.is_supported(intent_name):
                intents.append(Intent(intent_name, str(value).casefold()))
            else:
                logger.info(f"Unknown intent: {intent_name=} -> {value=}")

    logger.debug(f"Extracted {len(intents)} intents from the annotations")

    return intents


def _extract_intent_name(data: str) -> str:
    return "-".join(data.split("-")[2:]).casefold()
