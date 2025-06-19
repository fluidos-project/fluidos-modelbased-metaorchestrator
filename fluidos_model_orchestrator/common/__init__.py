from abc import ABC
from typing import Any

from fluidos_model_orchestrator.common.flavor import Flavor
from fluidos_model_orchestrator.common.intent import Intent
from fluidos_model_orchestrator.common.intent import requires_validation
from fluidos_model_orchestrator.common.model import ModelPredictRequest
from fluidos_model_orchestrator.common.model import ModelPredictResponse  # noqa
from fluidos_model_orchestrator.common.resource import ExternalResourceProvider
from fluidos_model_orchestrator.common.resource import Resource
from fluidos_model_orchestrator.common.resource import ResourceProvider
from fluidos_model_orchestrator.model import convert_to_model_request


class ResourceFinder(ABC):
    def find_best_match(self, resource: Resource, namespace: str) -> list[ResourceProvider]:
        raise NotImplementedError()

    def find_service(self, id: str, service: Intent, namespace: str) -> list[ExternalResourceProvider]:
        raise NotImplementedError()

    def retrieve_all_flavors(self, namespace: str) -> list[Flavor]:
        raise NotImplementedError()

    def update_local_flavor(self, flavor: Flavor, data: Any, namespace: str) -> None:
        raise NotImplementedError()


def requires_monitoring(spec: dict[str, Any], namespace: str | None) -> list[Intent]:
    if namespace is None:
        namespace = "default"
    request: ModelPredictRequest | None = convert_to_model_request(spec, namespace)

    if request is None:
        return []

    return [
        intent for intent in request.intents
        if requires_validation(intent)
    ]
