from typing import Any

import kopf
from kopf import PermanentError
from kopf import TemporaryError

from logging import Logger

from .model import convert_to_model_request
from .model import get_model_object
from .common import Intent, ModelInterface
from .common import ModelPredictRequest
from .common import ModelPredictResponse

from .start_and_stop import configure  # noqa
from .start_and_stop import cleanup_function  # noqa
from .healthz import healtz_get_current_timestamp  # noqa

from .deployment import deploy
from .resources import ResourceFinder
from .resources import ResourceProvider
from .resources import get_resource_finder


@kopf.on.create("fluidosdeployments")  # type: ignore
def creation_handler(spec: dict[str, Any], name: str, namespace: str, logger: Logger, errors: kopf.ErrorsMode = kopf.ErrorsMode.PERMANENT, **kwargs: str) -> None:
    logger.info("Processing incoming request")
    logger.debug(f"Received request: {spec}")

    request: ModelPredictRequest = convert_to_model_request(spec)

    if request is None:
        raise kopf.PermanentError("Request is not valid, discarding")

    predictor: ModelInterface = get_model_object(request)

    prediction: ModelPredictResponse = predictor.predict(request)

    if prediction is None:
        raise ValueError("Model unable to provide valid prediction")

    finder: ResourceFinder = get_resource_finder(request, prediction)

    best_match: ResourceProvider | None = finder.find_best_match(prediction.to_resource())

    # find other resources types based on the intents
    expanding_resources: list[tuple[ResourceProvider, Intent]] = _find_expanding_resources(finder, request.intents)

    if best_match is None:
        raise RuntimeError("Unable to find resource matching requirement")

    if not best_match.reserve():
        raise TemporaryError(f"Unable to find {best_match}")

    if not deploy(spec, best_match, expanding_resources):
        raise PermanentError("Unable to deploy")


def _find_expanding_resources(finder: ResourceFinder, intents: list[Intent]) -> list[tuple[ResourceProvider, Intent]]:
    resources_and_intents: list[tuple[ResourceProvider, Intent]] = list()

    for (resource, intent) in [
        (finder.find_best_match(intent), intent) for intent in intents if intent.has_external_requirement()
    ]:
        if resource is not None:
            resources_and_intents.append(
                (resource, intent)
            )

    return resources_and_intents
