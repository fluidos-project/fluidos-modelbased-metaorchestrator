from __future__ import annotations

import logging
from typing import Any

from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common.model import OrchestratorInterface


logger = logging.getLogger(__name__)


class Orchestrator(OrchestratorInterface):
    def load(self) -> Any:
        raise NotImplementedError("Not implemented: abstract method")

    def predict(self, data: ModelPredictRequest, architecture: str = "amd64") -> ModelPredictResponse | None:
        return None
