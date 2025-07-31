from logging import Logger
from typing import Any

import kopf  # type: ignore

from fluidos_model_orchestrator.common import ResourceFinder
from fluidos_model_orchestrator.common.intent import Intent
from fluidos_model_orchestrator.common.intent import KnownIntent
from fluidos_model_orchestrator.common.model import ModelPredictRequest
from fluidos_model_orchestrator.common.model import ModelPredictResponse
from fluidos_model_orchestrator.common.model import OrchestratorInterface
from fluidos_model_orchestrator.common.resource import ExternalResourceProvider
from fluidos_model_orchestrator.common.resource import ResourceProvider
from fluidos_model_orchestrator.configuration import CONFIGURATION
from fluidos_model_orchestrator.deployment import deploy
from fluidos_model_orchestrator.model import convert_to_model_request
from fluidos_model_orchestrator.model import get_model_object
from fluidos_model_orchestrator.resources import get_resource_finder
from fluidos_model_orchestrator.resources.mspl.mspl_resource_provider import MSPLIntentWrapper


@kopf.on.create("fluidosdeployments", )  # type: ignore
async def metaorchestration(spec: dict[str, Any], name: str, namespace: str, logger: Logger, errors: kopf.ErrorsMode = kopf.ErrorsMode.PERMANENT, **kwargs: str) -> dict[str, dict[str, ResourceProvider | list[str] | None | str] | str]:
    logger.info("Processing incoming request")
    logger.debug(f"Received request: {spec}")

    request: ModelPredictRequest | None = convert_to_model_request(spec, namespace)

    if request is None:
        logger.error("Request is not valid, discarding")
        return {
            "status": "Failure",
            "msg": "Invalid request"
        }

    predictor: OrchestratorInterface = get_model_object(request)

    prediction: ModelPredictResponse | None = predictor.predict(request, CONFIGURATION.architecture)  # this should use a system defined default, thus from the configuration

    if prediction is None:
        logger.error("Model unable to provide valid prediction")
        return {
            "status": "Failure",
            "msg": "Model unable to provide valid prediction"
        }
    else:
        logger.debug(f"Predicted resources for {spec['metadata']['name']}: {prediction}")

    finder: ResourceFinder = get_resource_finder(request, prediction)

    resources = finder.find_best_match(prediction.to_resource(), namespace)

    logger.debug(f"{resources=}")

    best_matches: list[ResourceProvider] = validate_with_intents(
        predictor.rank_resources(
            resources,
            prediction,
            request
        ), request.intents, logger)

    if not len(best_matches):
        logger.info("Unable to find resource matching requirement")

        return {
            "status": "Failure",
            "msg": "Unable to find resource matching requirement"
        }
    else:
        logger.info(f"Retrieved {len(best_matches)} valid resource providers")

    best_match = best_matches[0]

    logger.info(f"Selected {best_match.id} of type {type(best_match)}")

    if not best_match.acquire(namespace):
        logger.info(f"Unable to acquire {best_match}")

        return {
            "status": "Failure",
            "msg": "Unable to find resource matching requirement"
        }

    # find other resources types based on the intents
    expanding_resources: list[tuple[ExternalResourceProvider, Intent]] = _find_expanding_resources(finder, request.intents, name, namespace)

    if not await deploy(spec, best_match, expanding_resources, prediction, namespace):
        logger.info("Unable to deploy")

        return {
            "status": "Failure",
            "msg": "Unable to find resource matching requirement"
        }

    logger.info("Checking if monitoring is required")
    if CONFIGURATION.monitor_enabled:
        intents_to_monitor = [intent for intent in request.intents if intent.needs_monitoring()]
        if len(intents_to_monitor):
            # inform bastion
            pass

    return {
        "status": "Success",
        "deployed": {
            "resource_provider": str(best_match),
            "expandind_resources": [res[1].name.name for res in expanding_resources],
        }
    }


def validate_with_intents(providers: list[ResourceProvider], intents: list[Intent], logger: Logger) -> list[ResourceProvider]:
    valid_providers: list[ResourceProvider] = []

    logger.info(intents)

    for provider in providers:
        for intent in intents:
            if not intent.validates(provider):
                logger.info(f"{intent} does not validate {provider}")
                break
        else:
            logger.info(f"{provider} is validating all intents")
            valid_providers.append(provider)

    return valid_providers


def _find_expanding_resources(finder: ResourceFinder, intents: list[Intent], id: str, namespace: str) -> list[tuple[ExternalResourceProvider, Intent]]:
    resources_and_intents: list[tuple[ExternalResourceProvider, Intent]] = list()

    for (resources, intent) in [
        (finder.find_service(id, intent, namespace), intent) for intent in intents if intent.name == KnownIntent.service
    ]:
        if len(resources):
            resource: ExternalResourceProvider = resources[0]
            resources_and_intents.append(
                (resource, intent)
            )

    return resources_and_intents + [
        (MSPLIntentWrapper(intent), intent) for intent in intents if intent.name == KnownIntent.mspl
    ]
