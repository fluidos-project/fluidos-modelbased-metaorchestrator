from __future__ import annotations

import uuid
from collections.abc import Iterable

from fluidos_model_orchestrator.common import cpu_to_int
from fluidos_model_orchestrator.common import memory_to_int
from fluidos_model_orchestrator.common import ModelInterface
from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.common import ResourceProvider


class FluidosModelEnsemble(ModelInterface):
    def __init__(self, models: Iterable[ModelInterface]):
        self.models = list(models)

    def predict(self, data: ModelPredictRequest, architecture: str = "amd64") -> ModelPredictResponse:
        return _merge_prediction_responses(
            model.predict(data, architecture) for model in self.models
        )

    def rank_resources(self, providers: list[ResourceProvider], prediction: ModelPredictResponse) -> list[ResourceProvider]:
        for model in self.models:
            providers = model.rank_resource(providers, prediction)

        return providers


def _merge_prediction_responses(responses: Iterable[ModelPredictResponse]) -> ModelPredictResponse:
    merged_response: ModelPredictResponse | None = None

    for response in responses:
        if merged_response is None:
            merged_response = response
        else:
            merged_response.resource_profile = _merge_resource_profile(merged_response.resource_profile, response.resource_profile)

    if merged_response is None:
        id = str(uuid.uuid4())
        return ModelPredictResponse(id=id, resource_profile=Resource(id=id))

    return merged_response


def _merge_resource_profile(res1: Resource, res2: Resource) -> Resource:
    cpu: str | None

    if res1.cpu is not None:
        if res2.cpu is not None:
            cpu = res1.cpu if cpu_to_int(res1.cpu) > cpu_to_int(res2.cpu) else res2.cpu
        else:
            cpu = res1.cpu
    else:
        cpu = res2.cpu

    memory: str | None

    if res1.memory is not None:
        if res2.memory is not None:
            memory = res1.memory if memory_to_int(res1.memory) > memory_to_int(res2.memory) else res2.memory
        else:
            memory = res1.memory
    else:
        memory = res2.memory

    return Resource(
        id=res1.id,
        cpu=cpu,
        memory=memory,
        architecture=res1.architecture,
        gpu=res1.gpu if res1.gpu is not None else res2.gpu,
        ephemeral_storage=res1.ephemeral_storage if res1.ephemeral_storage is not None else res2.ephemeral_storage,
        persistent_storage=res1.persistent_storage if res1.persistent_storage is not None else res2.persistent_storage,
        region=res1.region if res1.region is not None else res2.region,
    )
