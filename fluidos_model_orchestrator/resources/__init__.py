from __future__ import annotations

import logging

from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import ResourceFinder
from fluidos_model_orchestrator.resources.rear import REARResourceFinder
from fluidos_model_orchestrator.rob_finder import ROBResourceFinder


logger = logging.getLogger(__name__)


def get_resource_finder(request: ModelPredictRequest | None = None, predict: ModelPredictResponse | None = None) -> ResourceFinder:
    if predict is not None and predict.id.startswith("rob-"):
        logger.info("ROBResourceFinder being returned")
        return ROBResourceFinder()
    logger.info("REARResourceFinder being returned")
    return REARResourceFinder()
