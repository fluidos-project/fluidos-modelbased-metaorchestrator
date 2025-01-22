import logging
from typing import Any

from ..common import Intent
from ..common import KnownIntent
from ..common import ModelPredictRequest
from ..common import OrchestratorInterface
from ..container import extract_image_embedding
from .candidate_generation.model import Orchestrator as CandidateGenerator
from .carbon_aware.orchestrator import CarbonAwareOrchestrator
from .ensemble import FluidosModelEnsemble
from fluidos_model_orchestrator.model.utils import FLUIDOS_COL_NAMES
# from .model_basic_ranker.model import Orchestrator as BasicRanker

logger = logging.getLogger(__name__)


_model_instances: dict[type[OrchestratorInterface], OrchestratorInterface] = {}

_model_characteristics: list[tuple[set[KnownIntent], type[OrchestratorInterface]]] = [
    ({
        KnownIntent.architecture,
        KnownIntent.compliance,
        KnownIntent.cpu,
        KnownIntent.gpu,
        KnownIntent.latency,
        KnownIntent.location,
        KnownIntent.memory,
        KnownIntent.throughput,
    }, CandidateGenerator),
    # ({KnownIntent.latency, KnownIntent.location, KnownIntent.memory, KnownIntent.cpu}, BasicRanker),
    ({KnownIntent.carbon_aware, KnownIntent.max_delay}, CarbonAwareOrchestrator)
]


def _is_compatible(request: set[KnownIntent], model: set[KnownIntent]) -> bool:
    return 0 != len(model & request)


def _get_model(model_type: type[OrchestratorInterface]) -> OrchestratorInterface:
    if model_type not in _model_instances:
        _model_instances[model_type] = model_type()

    return _model_instances[model_type]


def get_model_object(request: ModelPredictRequest) -> OrchestratorInterface:
    logger.info(f"Retrieving model interface for {request.id}")

    request_intent_signature = {intent.name for intent in request.intents}

    matching_models = [
        name for (model_signature, name) in _model_characteristics if _is_compatible(request_intent_signature, model_signature)
    ]

    logger.info(f"{len(matching_models)}")

    if 1 == len(matching_models):
        logger.info(f"Returning model {matching_models[0]}")
        return _get_model(matching_models[0])
    elif 1 < len(matching_models):
        logger.info(f"Returning an ensemble of the models {matching_models}")
        return FluidosModelEnsemble(
            _get_model(model_name) for model_name in matching_models
        )

    logger.debug("No matching models, returning what? Dummy? or CG as default?")
    return _get_model(CandidateGenerator)


def convert_to_model_request(spec: Any, namespace: str) -> ModelPredictRequest | None:
    logger.info("Converting incoming custom resource to model request")

    request: ModelPredictRequest | None = None

    if spec["kind"] == "Deployment":
        logger.debug("Processing Deployment object")
        intents = _extract_intents(spec["metadata"].get("annotations", {}))
        for container in spec["spec"]["template"]["spec"]["containers"]:
            pass
            # intents.extend(
            #     _extract_resource_intents(
            #         container.get("resources", {}).get("requests", {})
            #     )
            # )

        request = ModelPredictRequest(
            id=spec["metadata"]["name"],
            namespace=namespace,
            pod_request={
                FLUIDOS_COL_NAMES.POD_MANIFEST: spec["spec"]["template"]
            },
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
            pod_request={
                FLUIDOS_COL_NAMES.POD_MANIFEST: spec
            },
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
    logger.debug("Extracting intents from annotations")
    intents = [
        Intent(KnownIntent.get_intent(key), str(value)) for key, value in annotations.items() if KnownIntent.is_supported(key.casefold())
    ]

    logger.debug(f"Extracted {len(intents)} intents from the annotations")

    return intents
