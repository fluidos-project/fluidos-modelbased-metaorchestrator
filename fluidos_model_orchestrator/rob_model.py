from __future__ import annotations

import logging
import uuid
from typing import Any

from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common.model import OrchestratorInterface
from fluidos_model_orchestrator.common.resource import Resource
from fluidos_model_orchestrator.configuration import CONFIGURATION


logger = logging.getLogger(__name__)


class Orchestrator(OrchestratorInterface):
    def load(self) -> Any:
        raise NotImplementedError("Not implemented: abstract method")

    def predict(self, data: ModelPredictRequest, architecture: str = "amd64") -> ModelPredictResponse | None:
        id = "rob-" + uuid.uuid4().hex

        return ModelPredictResponse(
            id=id,
            resource_profile=Resource(
                id=id,
                architecture=CONFIGURATION.architecture
            ),
            delay=0
        )
