from logging import Logger
from typing import Any

import kopf  # type: ignore
from kopf import PermanentError

from .common import find_best_validation
from .common import Intent
from .common import ModelInterface
from .common import ModelPredictRequest
from .common import ModelPredictResponse
from .common import ResourceFinder
from .common import ResourceProvider
from .common import validate_on_intent
from .deployment import deploy
from .healthz import healtz_get_current_timestamp  # noqa
from .model import convert_to_model_request
from .model import get_model_object
from .rescheduler import rescheduler  # noqa
from .resources import get_resource_finder
from .start_and_stop import cleanup_function  # noqa
from .start_and_stop import configure  # noqa


@kopf.on.create("fluidosdeployments")
def creation_handler(spec: dict[str, Any], name: str, namespace: str, logger: Logger, errors: kopf.ErrorsMode = kopf.ErrorsMode.PERMANENT, **kwargs: str) -> dict[str, dict[str, ResourceProvider | list[tuple[ResourceProvider, Intent]] | None] | str]:
    logger.info("Processing incoming request")
    logger.debug(f"Received request: {spec}")

    request: ModelPredictRequest | None = convert_to_model_request(spec, namespace)

    if request is None:
        logger.error("Request is not valid, discarding")
        return {
            "msg": "Invalid request"
        }

    predictor: ModelInterface = get_model_object(request)

    prediction: ModelPredictResponse = predictor.predict(request, "amd64")

    if prediction is None:
        logger.error("Model unable to provide valid prediction")
        return {
            "msg": "Model unable to provide valid prediction"
        }
    else:
        logger.debug(f"Predicted resources for {spec['metadata']['name']}: {prediction}")

    finder: ResourceFinder = get_resource_finder(request, prediction)

    best_match: ResourceProvider | None = find_best_validation(finder.find_best_match(prediction.to_resource(), namespace), request.intents)

    # find other resources types based on the intents
    expanding_resources: list[tuple[ResourceProvider, Intent]] = _find_expanding_resources(finder, request.intents, namespace)

    if best_match is None:
        logger.error("Unable to find resource matching requirement")

        return {
            "msg": "Unable to find resource matching requirement"
        }

    if not best_match.acquire():
        raise PermanentError(f"Unable to acquire {best_match}")

    if not deploy(spec, best_match, expanding_resources):
        raise PermanentError("Unable to deploy")

    return {
        "deployed": {
            "resource_provider": best_match,
            "expandind_resources": expanding_resources or None,
            "validation": []
        },
        "msg": "Successful"
    }


def _find_expanding_resources(finder: ResourceFinder, intents: list[Intent], namespace: str) -> list[tuple[ResourceProvider, Intent]]:
    resources_and_intents: list[tuple[ResourceProvider, Intent]] = list()

    for (resources, intent) in [
        (finder.find_best_match(intent, namespace), intent) for intent in intents if intent.has_external_requirement()
    ]:
        if len(resources):
            resource: ResourceProvider = validate_on_intent(resources, intent)
            resources_and_intents.append(
                (resource, intent)
            )

    return resources_and_intents
