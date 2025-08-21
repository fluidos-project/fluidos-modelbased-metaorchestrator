from __future__ import annotations

import logging
import uuid
from typing import Any

from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common.intent import Intent
from fluidos_model_orchestrator.common.model import OrchestratorInterface
from fluidos_model_orchestrator.common.resource import ExternalResourceProvider
from fluidos_model_orchestrator.common.resource import Resource
from fluidos_model_orchestrator.common.resource import ResourceProvider
from fluidos_model_orchestrator.configuration import CONFIGURATION
from fluidos_model_orchestrator.resources.rear.finder import REARResourceFinder


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


class ROBResourceFinder(REARResourceFinder):
    def find_best_match(self, resource: Resource, namespace: str, solver_name: str | None = None) -> list[ResourceProvider]:
        logger.info("Returning only local at the beginning, peering already in place")
        resources = self._find_local(resource=resource, namespace=namespace)

        logger.info("%d flavors found", len(resources))
        return resources

    def find_service(self, id: str, service: Intent, namespace: str) -> list[ExternalResourceProvider]:
        logger.info("Not supported")
        return []
