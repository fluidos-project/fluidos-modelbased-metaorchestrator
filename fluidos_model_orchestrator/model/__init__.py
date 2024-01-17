from typing import Any

from ..common import ModelInterface, ModelPredictResponse
from ..common import ModelPredictRequest
from ..common import Intent

from kopf import PermanentError

import logging

logger = logging.getLogger(__name__)


class DummyOrchestrator(ModelInterface):
    def predict(self, req: ModelPredictRequest) -> ModelPredictResponse:
        return None


def get_model_object() -> ModelInterface:
    logger.info("Retrieving model interface")
    return DummyOrchestrator()


def convert_to_model_request(spec: Any) -> ModelPredictRequest:
    logger.info("Converting incoming custom resource to model request")
    if spec["kind"] == "Deployment":
        logger.debug("Processing Deployment object")
        return ModelPredictRequest(id=f"{spec['metadata']['name']}", pod_request=spec["spec"]["template"], intents=_extract_intents(spec["metadata"].get("annotations", {})))

    if spec["kind"] == "Pod":
        logger.debug("Processing Pod object")
        return ModelPredictRequest(id=spec["metadata"]["name"], pod_request=spec, intents=_extract_intents(spec["metadata"].get("annotations", {})))

    logger.error(f"Unsupported kind {spec['kind']}")
    raise PermanentError(f"Unsupported kind {spec['kind']}")


def _extract_intents(annotations: dict[str, str]) -> list[Intent]:
    logger.debug("Extracting intens from annotations")
    intents = [
        Intent(_extract_intent_name(key).casefold(), str(value).casefold()) for key, value in annotations.items() if key.casefold().startswith("fluidos-intent-")
    ]

    logger.debug(f"Extracted {len(intents)} intents from the annotations")

    return intents


def _extract_intent_name(data: str) -> str:
    return "-".join(data.split("-")[2:])
