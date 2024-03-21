from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from enum import Enum
from enum import auto
from enum import unique

import logging


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Resource:
    id: str
    cpu: str | None = None
    memory: str | None = None
    architecture: str | None = None
    gpu: str | None = None
    ephemeral_storage: str | None = None
    persistent_storage: str | None = None
    region: str | None = None

    def can_run_on(self, flavour: Flavour) -> bool:
        logger.debug(f"Testing {self=} against {flavour=}")
        if not _cpu_compatible(self.cpu, flavour.cpu):
            return False

        if not _memory_compatible(self.memory, flavour.memory):
            return False

        if self.architecture is not None and self.architecture != flavour.architecture:
            return False

        if self.gpu is not None and int(self.gpu) > int(flavour.gpu):
            return False

        # TODO: add checks for storage

        return True


def _memory_compatible(memory_a_spec: str, memory_b_spec: str) -> bool:
    memory_a: int = _memory_to_int(memory_a_spec)
    memory_b: int = _memory_to_int(memory_b_spec)

    return memory_b >= memory_a


def _memory_to_int(spec: str) -> int:
    unit = spec[-2:]
    magnitude = int(spec[:-2])

    if unit == "Ki":
        return magnitude
    if unit == "Mi":
        return 1024 * magnitude
    if unit == "Gi":
        return 1024 * 1024 * magnitude

    raise ValueError(f"Not known {unit=}")


def _cpu_compatible(cpu_a_spec: str, cpu_b_spec: str) -> bool:
    cpu_a: int = _cpu_to_int(cpu_a_spec)
    cpu_b: int = _cpu_to_int(cpu_b_spec)

    return cpu_b >= cpu_a


def _cpu_to_int(spec: str) -> int:
    if spec[-1] == 'n':
        return int(spec[:-1])
    elif spec[-1] == "m":
        return int(spec[:-1]) * 1000
    else:
        return int(spec) * (1000 * 1000)


@dataclass
class Flavour:
    id: str
    cpu: str
    architecture: str
    gpu: str
    memory: str


@dataclass
class ContainerImageEmbedding:
    image: str
    embedding: str | None = None


@dataclass
class ModelPredictRequest:
    id: str
    pod_request: Any
    container_image_embeddings: list[ContainerImageEmbedding]
    intents: list[Intent] = field(default_factory=list)


@dataclass
class ModelPredictResponse:
    id: str
    resource_profile: Resource

    def to_resource(self) -> Resource:
        return self.resource_profile


class ModelInterface(ABC):
    @abstractmethod
    def predict(self, data: ModelPredictRequest) -> ModelPredictResponse:
        raise NotImplementedError("Not implemented: abstract method")


@unique
class KnownIntent(Enum):
    # k8s resources
    cpu = auto()
    memory = auto()

    # high order requests
    latency = auto()
    location = auto()
    throughput = auto()
    compliance = auto()
    energy = auto()
    battery = auto()

    # service
    service = auto()

    def __repr__(self) -> str:
        return super().__repr__()

    def to_intent_key(self) -> str:
        return f"fluidos-intent-{self.name}"

    @staticmethod
    def is_supported(intent_name: str) -> bool:
        intent_name = "-".join(intent_name.split("-")[2:]).casefold()

        return any(
            known_intent.name == intent_name for known_intent in KnownIntent
        )

    @staticmethod
    def get_intent(intent_name: str) -> KnownIntent:
        # defensive programming
        if not KnownIntent.is_supported(intent_name):
            raise ValueError(f"Unsupported intent: {intent_name=}")

        name = "-".join(intent_name.split("-")[2:]).casefold()
        return next(known_intent for known_intent in KnownIntent if known_intent.name == name)


@dataclass
class Intent:
    name: KnownIntent
    value: str


@dataclass
class Configuration:
    local_node_key: str = "fluidos.eu/resource-node"
    remote_node_key: str = "liqo.io/remote-cluster-id"


CONFIGURATION = Configuration()
