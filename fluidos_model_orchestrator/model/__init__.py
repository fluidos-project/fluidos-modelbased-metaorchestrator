import logging
from typing import Any

from ..common import Intent
from ..common import KnownIntent
from ..common import OrchestratorInterface
from ..common import ModelPredictRequest
from ..container import extract_image_embedding
from .candidate_generation import Orchestrator as CG
from .dummy import DummyOrchestrator
from .ensemble import FluidosModelEnsemble
# from .two_tower_v1.orchestrator import TwoTowerOrchestrator


logger = logging.getLogger(__name__)


_model_instances: dict[str, OrchestratorInterface] = {
    "CG": CG(),
    # "2T": TwoTowerOrchestrator(),
    "dummy": DummyOrchestrator()
}

_model_characteristics: list[tuple[set[KnownIntent], str]] = [
    ({known_intent for known_intent in KnownIntent}, "CG"),
    # ({KnownIntent.latency, KnownIntent.location, KnownIntent.memory, KnownIntent.cpu}, "2T")
]


def _is_subset(s1: set[KnownIntent], s2: set[KnownIntent]) -> bool:
    for e1 in s1:
        for e2 in s2:
            if e1 == e2:
                break
        else:
            return False

    return True


def get_model_object(request: ModelPredictRequest) -> OrchestratorInterface:
    logger.info(f"Retrieving model interface for {request}")

    request_intent_signature = {intent.name for intent in request.intents}

    matching_models = [
        name for (model_signature, name) in _model_characteristics if _is_subset(request_intent_signature, model_signature)
    ]

    """
    if 1 == len(matching_models):
        logger.debug(f"Returning model {matching_models[0]}")
        return _model_instances[matching_models[0]]
    elif 1 < len(matching_models):
        logger.debug(f"Regurning an ensemble of the models {matching_models}")
        return FluidosModelEnsemble(
            _model_instances[model_name] for model_name in matching_models
        )
    """
    return _model_instances["dummy"]



def convert_to_model_request(spec: Any, namespace: str) -> ModelPredictRequest | None:
    logger.info("Converting incoming custom resource to model request")

    request: ModelPredictRequest | None = None

    if spec["kind"] == "Deployment":
        logger.debug("Processing Deployment object")
        intents = _extract_intents(spec["metadata"].get("annotations", {}))
        for container in spec["spec"]["template"]["spec"]["containers"]:
            intents.extend(
                _extract_resource_intents(
                    container.get("resources", {}).get("requests", {})
                )
            )

        request = ModelPredictRequest(
            id=spec["metadata"]["name"],
            namespace=namespace,
            pod_request=spec["spec"]["template"],
            intents=intents,
            container_image_embeddings=[extract_image_embedding(container["image"]) for container in spec["spec"]["template"]["spec"]["containers"]]
        )

    if spec["kind"] == "Pod":
        logger.debug("Processing Pod object")
        intents = _extract_intents(spec["metadata"].get("annotations", {}))

        for container in spec["spec"]["containers"]:
            intents.extend(
                _extract_resource_intents(
                    container.get("resources", {}).get("requests", {})
                )
            )

        request = ModelPredictRequest(
            id=spec["metadata"]["name"],
            namespace=namespace,
            pod_request=spec,
            intents=intents,
            container_image_embeddings=[extract_image_embedding(container["image"]) for container in spec["spec"]["containers"]]
        )

    if request is not None:
        return request

    logger.error(f"Unsupported kind {spec['kind']}")
    return None


def _extract_resource_intents(requests: dict[str, str]) -> list[Intent]:
    logger.debug("Extracting resources intents from request")

    return [
        Intent(key[1], requests[key[0]]) for key in [("cpu", KnownIntent.cpu), ("memory", KnownIntent.memory)] if key[0] in requests
    ]


def _extract_intents(annotations: dict[str, str]) -> list[Intent]:
    logger.debug("Extracting intens from annotations")
    intents = [
        Intent(KnownIntent.get_intent(key), str(value).casefold()) for key, value in annotations.items() if KnownIntent.is_supported(key.casefold())
    ]

    logger.debug(f"Extracted {len(intents)} intents from the annotations")

    return intents
