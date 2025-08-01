from __future__ import annotations

import json
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import cast

from fluidos_model_orchestrator.common.flavor import Flavor
from fluidos_model_orchestrator.common.flavor import FlavorK8SliceData
from fluidos_model_orchestrator.common.flavor import FlavorType
from fluidos_model_orchestrator.common.flavor import GPUData


def _memory_compatible(req_spec: str | None, offer_spec: str | None) -> bool:
    if req_spec is None:
        return False
    if offer_spec is None:
        return False

    memory_a: int = memory_to_int(req_spec)
    memory_b: int = memory_to_int(offer_spec)

    return memory_b >= memory_a


def memory_to_int(spec: str) -> int:
    unit = spec[-2:]
    magnitude = int(spec[:-2])

    if unit == "Ki":
        return magnitude
    if unit == "Mi":
        return 1024 * magnitude
    if unit == "Gi":
        return 1024 * 1024 * magnitude

    raise ValueError(f"Not known {unit=}")


def _convert_to_gpudata(gpu: GPUData | str | int | dict[str, Any]) -> GPUData:
    if type(gpu) is GPUData:
        return gpu
    if type(gpu) is dict:
        # loads of faith required here:
        return GPUData(
            cores=gpu.get("core", "0"),
            memory=gpu.get("memory", "0"),
            model=gpu.get("model", "")
        )

    if type(gpu) is str:
        gpu = int(gpu)
    if type(gpu) is int:
        return GPUData(cores=gpu)

    raise ValueError(f"Unable to convert value '{gpu}' of type {type(gpu)}")


def _check_gpu(provider: ResourceProvider, value: str) -> bool:
    if provider.flavor.spec.flavor_type.type_identifier is FlavorType.K8SLICE:
        data = json.loads(value)

        gpu: GPUData = _convert_to_gpudata(cast(FlavorK8SliceData, provider.flavor.spec.flavor_type.type_data).characteristics.gpu)

        if type(data) is dict:
            return gpu.can_run_on(_convert_to_gpudata(data))
        elif type(data) is int:
            return int(gpu.cores) >= data

    return False


def _cpu_compatible(req_spec: str | None, offer_spec: str | None) -> bool:
    if req_spec is None:
        return False
    if offer_spec is None:
        return False
    cpu_a: int = cpu_to_int(req_spec)
    cpu_b: int = cpu_to_int(offer_spec)

    return cpu_b >= cpu_a


def cpu_to_int(spec: str) -> int:
    if spec[-1] == "n":
        return int(spec[:-1])
    elif spec[-1] == "m":
        return int(spec[:-1]) * 1000
    else:
        return int(spec) * (1000 * 1000)


class ResourceProvider(ABC):
    def __init__(self, id: str, flavor: Flavor) -> None:
        self.id = id
        self.flavor = flavor

    @abstractmethod
    def acquire(self, namespace: str) -> bool:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def get_label(self) -> dict[str, str]:
        raise NotImplementedError("Abstract method")

    # def __str__(self) -> str:
    #     return f"{self.id=}: {self.flavor=}"

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        raise NotImplementedError("Abstract method")


class ExternalResourceProvider(ABC):
    def enrich(self, container: dict[str, Any], name: str) -> None:
        raise NotImplementedError("Abstract method")


@dataclass(kw_only=True)
class Resource:
    id: str
    cpu: str | None = None
    memory: str | None = None
    architecture: str | None = None
    gpu: str | None = None
    storage: str | None = None
    region: str | None = None
    pods: str | int | None = None

    def can_run_on(self, flavor: Flavor) -> bool:
        if flavor.spec.flavor_type.type_identifier is not FlavorType.K8SLICE:
            return False

        type_data = cast(FlavorK8SliceData, flavor.spec.flavor_type.type_data)

        if self.cpu is not None and not _cpu_compatible(self.cpu, type_data.characteristics.cpu):
            return False

        if self.memory is not None and not _memory_compatible(self.memory, type_data.characteristics.memory):
            return False

        if self.architecture is not None and self.architecture != type_data.characteristics.architecture:
            return False

        if self.gpu is not None:
            our_gpu = _convert_to_gpudata(self.gpu)
            their_gpu = _convert_to_gpudata(type_data.characteristics.gpu)

            return our_gpu.can_run_on(their_gpu)

        if self.pods is not None:
            return int(self.pods) == int(type_data.characteristics.pods if type_data.characteristics.pods is not None else 0)

        # TODO: add checks for storage

        return True
