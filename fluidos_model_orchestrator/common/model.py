from abc import ABC
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from fluidos_model_orchestrator.common.intent import Intent
from fluidos_model_orchestrator.common.resource import Resource
from fluidos_model_orchestrator.common.resource import ResourceProvider


@dataclass
class ContainerImageEmbedding:
    image: str
    embedding: str | None = None


@dataclass(kw_only=True)
class ModelPredictRequest:
    id: str
    namespace: str
    pod_request: Any
    container_image_embeddings: list[ContainerImageEmbedding]
    intents: list[Intent] = field(default_factory=list)


@dataclass
class ModelPredictResponse:
    id: str
    resource_profile: Resource
    delay: int = 0  # time in hours

    def to_resource(self) -> Resource:
        return self.resource_profile


class OrchestratorInterface(ABC):
    def load(self) -> Any:
        raise NotImplementedError("Not implemented: abstract method")

    def predict(self, data: ModelPredictRequest, architecture: str = "amd64") -> ModelPredictResponse | None:
        return None

    def rank_resources(self, providers: list[ResourceProvider], prediction: ModelPredictResponse, request: ModelPredictRequest) -> list[ResourceProvider]:
        return providers
