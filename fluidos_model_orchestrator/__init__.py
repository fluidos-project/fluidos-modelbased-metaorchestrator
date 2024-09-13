from logging import Logger
from typing import Any

import kopf  # type: ignore

from .common import Intent
from .common import ModelPredictRequest
from .common import ModelPredictResponse
from .common import OrchestratorInterface
from .common import ResourceFinder
from .common import ResourceProvider
from .common import validate_on_intent
from .configuration import CONFIGURATION
from .daemons_and_times.flavor import daemons_for_flavours_observation  # noqa
from .deployment import deploy
from .healthz import healtz_get_current_timestamp  # noqa
from .model import convert_to_model_request
from .model import get_model_object
from .resources import get_resource_finder
from .start_and_stop import cleanup_function  # noqa
from .start_and_stop import configure  # noqa
# from .rescheduler import rescheduler


@kopf.on.create("fluidosdeployments")  # type: ignore
async def creation_handler(spec: dict[str, Any], name: str, namespace: str, logger: Logger, errors: kopf.ErrorsMode = kopf.ErrorsMode.PERMANENT, **kwargs: str) -> dict[str, dict[str, ResourceProvider | list[tuple[ResourceProvider, Intent]] | None | str] | str]:
    logger.info("Processing incoming request")
    logger.debug(f"Received request: {spec}")

    request: ModelPredictRequest | None = convert_to_model_request(spec, namespace)

    if request is None:
        logger.error("Request is not valid, discarding")
        return {
            "msg": "Invalid request"
        }

    predictor: OrchestratorInterface = get_model_object(request)

    prediction: ModelPredictResponse | None = predictor.predict(request, CONFIGURATION.architecture)  # this should use a system defined default, thus from the configuration

    if prediction is None:
        logger.error("Model unable to provide valid prediction")
        return {
            "msg": "Model unable to provide valid prediction"
        }
    else:
        logger.debug(f"Predicted resources for {spec['metadata']['name']}: {prediction}")

    finder: ResourceFinder = get_resource_finder(request, prediction)

    resources = finder.find_best_match(prediction.to_resource(), namespace)

    logger.debug(f"{resources=}")

    best_matches: list[ResourceProvider] = validate_with_intents(
        predictor.rank_resource(
            resources,
            prediction,
            request
        ), request.intents)

    if not len(best_matches):
        logger.info("Unable to find resource matching requirement")

        return {
            "status": "Failure",
            "msg": "Unable to find resource matching requirement"
        }

    best_match = best_matches[0]

    if not best_match.acquire():
        logger.info(f"Unable to acquire {best_match}")

        return {
            "status": "Failure",
            "msg": "Unable to find resource matching requirement"
        }

    # find other resources types based on the intents
    expanding_resources: list[tuple[ResourceProvider, Intent]] = _find_expanding_resources(finder, request.intents, namespace)

    if not await deploy(spec, best_match, expanding_resources, prediction):
        logger.info("Unable to deploy")

        return {
            "status": "Failure",
            "msg": "Unable to find resource matching requirement"
        }

    return {
        "status": "Success",
        "deployed": {
            "resource_provider": str(best_match),
            "expandind_resources": expanding_resources or None,
            "validation": []
        }
    }


def validate_with_intents(providers: list[ResourceProvider], intents: list[Intent]) -> list[ResourceProvider]:
    return [
        provider for provider in providers if all(
            intent.validates(provider) for intent in intents
        )
    ]


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
