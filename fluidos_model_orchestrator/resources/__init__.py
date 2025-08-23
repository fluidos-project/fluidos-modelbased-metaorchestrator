from __future__ import annotations

import logging

from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import ResourceFinder
from fluidos_model_orchestrator.configuration import CONFIGURATION
from fluidos_model_orchestrator.resources.rear import REARResourceFinder


logger = logging.getLogger(__name__)


def get_resource_finder(request: ModelPredictRequest | None = None, predict: ModelPredictResponse | None = None) -> ResourceFinder:
    logger.info("REARResourceFinder being returned")
    return REARResourceFinder(CONFIGURATION)
