from typing import Any

import kopf
from kopf import PermanentError

from logging import Logger

from .model import convert_to_model_request
from .model import get_model_object
from .common import ModelInterface
from .common import ModelPredictRequest
from .common import ModelPredictResponse

from .start_and_stop import configure  # noqa
from .start_and_stop import cleanup_function  # noqa
from .healthz import healtz_get_current_timestamp  # noqa

from .deployment import deploy
from .resources import ResourceFinder
from .resources import ResourceProvider
from .resources import get_resource_finder


@kopf.on.create("modelbaseddeployment")
def creation_handler(spec: dict[str, Any], name: str, namespace: str, logger: Logger, errors=kopf.ErrorsMode.PERMANENT, **kwargs):
    logger.info("Processing incoming request")
    logger.debug(f"Received request: {spec}")

    predictor: ModelInterface = get_model_object()
    finder: ResourceFinder = get_resource_finder()

    request: ModelPredictRequest = convert_to_model_request(spec)

    if request is None:
        raise kopf.PermanentError("Request is not valid, discarding")

    prediction: ModelPredictResponse = predictor.predict(request)

    if prediction is None:
        raise ValueError("Model unable to provide valid prediction")

    best_match: ResourceProvider = finder.find_best_match(prediction.to_resource())

    if best_match is None:
        raise RuntimeError("Unable to find resource matching requirement")

    if not best_match.reserve():
        raise RuntimeError(f"Unable to find {best_match}")

    if not deploy(spec, best_match):
        raise PermanentError("Unable to deploy")
