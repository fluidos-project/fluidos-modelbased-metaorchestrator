from __future__ import annotations

import logging

from ..common import ModelPredictRequest
from ..common import ModelPredictResponse
from ..common import ResourceFinder
from fluidos_model_orchestrator.resources.rear import REARResourceFinder


logger = logging.getLogger(__name__)


def get_resource_finder(request: ModelPredictRequest | None, predict: ModelPredictResponse | None) -> ResourceFinder:
    logger.info("REARResourceFinder being returned")
    return REARResourceFinder()
