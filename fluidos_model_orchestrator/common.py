from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from enum import auto
from enum import Enum
from enum import unique
from typing import Any


logger = logging.getLogger(__name__)


class ResourceProvider(ABC):
    def __init__(self, id: str, flavor: Flavor) -> None:
        self.id = id
        self.flavor = flavor

    def acquire(self) -> bool:
        return True

    @abstractmethod
    def get_label(self) -> str:
        raise NotImplementedError("Abstract method")


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

    def can_run_on(self, flavor: Flavor) -> bool:
        logger.debug(f"Testing {self=} against {flavor=}")
        if not _cpu_compatible(self.cpu, flavor.characteristics.cpu):
            return False

        if not _memory_compatible(self.memory, flavor.characteristics.memory):
            return False

        if self.architecture is not None and self.architecture != flavor.characteristics.architecture:
            return False

        if self.gpu is not None and int(self.gpu) > int(flavor.characteristics.gpu):
            return False

        # TODO: add checks for storage

        return True


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


def _cpu_compatible(req_spec: str | None, offer_spec: str | None) -> bool:
    if req_spec is None:
        return False
    if offer_spec is None:
        return False
    cpu_a: int = cpu_to_int(req_spec)
    cpu_b: int = cpu_to_int(offer_spec)

    return cpu_b >= cpu_a


def cpu_to_int(spec: str) -> int:
    if spec[-1] == 'n':
        return int(spec[:-1])
    elif spec[-1] == "m":
        return int(spec[:-1]) * 1000
    else:
        return int(spec) * (1000 * 1000)


@dataclass
class FlavorCharacteristics:
    cpu: str
    architecture: str
    gpu: str
    memory: str
    pods: str | None = None
    ephemeral_storage: str | None = None
    persistent_storage: str | None = None


@unique
class FlavorType(Enum):
    K8SLICE = auto()
    VM = auto()
    SERVICE = auto()
    SENSOR = auto()

    @staticmethod
    def factory(type_name: str) -> FlavorType:
        if type_name == "k8s-fluidos":
            return FlavorType.K8SLICE

        raise ValueError(f"Not supported {type_name=}")


@dataclass
class Flavor:
    id: str
    type: FlavorType
    characteristics: FlavorCharacteristics
    owner: dict[str, Any]
    providerID: str
    policy: dict[str, Any] = field(default_factory=dict)
    optional_fields: dict[str, Any] = field(default_factory=dict)
    price: dict[str, Any] = field(default_factory=dict)
    location: dict[str, Any] = field(default_factory=dict)


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

    @abstractmethod
    def predict(self, data: ModelPredictRequest, architecture: str = "arm64") -> ModelPredictResponse | None:
        raise NotImplementedError("Not implemented: abstract method")

    def rank_resource(self, providers: list[ResourceProvider], prediction: ModelPredictResponse, request: ModelPredictRequest) -> list[ResourceProvider]:
        return providers


_always_true: Callable[[ResourceProvider, str], bool] = lambda provider, value: True


def _validate_regulations(provider: ResourceProvider, value: str) -> bool:
    """
    Assumes values of the form:
    GDPR
    DORA
    or such
    """
    value = value.casefold()
    for field_name, field_value in provider.flavor.optional_fields.items():
        if str(field_value).casefold() == value:
            return True

    return False


@unique
class KnownIntent(Enum):
    # k8s resources
    cpu = "cpu", False, lambda provider, value: _cpu_compatible(value, provider.flavor.characteristics.cpu)
    memory = "memory", False, lambda provider, value: _memory_compatible(value, provider.flavor.characteristics.memory)
    gpu = "gpu", False, lambda provider, value: int(value) <= int(provider.flavor.characteristics.gpu)
    architecture = "architecture", False, lambda provider, value: value == provider.flavor.characteristics.architecture

    # high order requests
    latency = "latency", False, _always_true
    location = "location", False, _always_true
    throughput = "throughput", False, _always_true
    compliance = "compliance", False, _validate_regulations
    energy = "energy", False, _always_true
    battery = "battery", False, _always_true

    # carbon aware requests
    deadline = "deadline", False, _always_true

    # service
    service = "service", True, _always_true

    def __new__(cls, *args: str, **kwds: str) -> KnownIntent:
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, external: bool, validator: Callable[[ResourceProvider, str], bool]):
        self._external = external
        self._validator = validator

    def __repr__(self) -> str:
        return super().__repr__()

    def to_intent_key(self) -> str:
        return f"fluidos-intent-{self.name}"

    def has_external_requirement(self) -> bool:
        return self._external

    def validates(self, provider: ResourceProvider, value: str) -> bool:
        return self._validator(provider, value)

    @staticmethod
    def is_supported(intent_name: str) -> bool:
        if intent_name.startswith("fluidos-intent-"):
            intent_name = "-".join(intent_name.split("-")[2:])

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

    def has_external_requirement(self) -> bool:
        return self.name.has_external_requirement()

    def validates(self, provider: ResourceProvider) -> bool:
        return self.name.validates(provider, self.value)


def validate_on_intent(resources: list[ResourceProvider], intent: Intent) -> ResourceProvider:
    return resources[0]  # for now


class ResourceFinder(ABC):
    def find_best_match(self, resource: Resource | Intent, namespace: str) -> list[ResourceProvider]:
        raise NotImplementedError()

    def retrieve_all_flavors(self, namespace: str) -> list[Flavor]:
        raise NotImplementedError()

    def update_local_flavor(self, flavor: Flavor, data: Any, namespace: str) -> None:
        raise NotImplementedError()


def build_flavor(flavor: dict[str, Any]) -> Flavor:
    return Flavor(
        id=flavor["metadata"]["name"],
        type=FlavorType.factory(flavor["spec"]["type"]),
        providerID=flavor["spec"]["providerID"],
        characteristics=FlavorCharacteristics(
            cpu=flavor["spec"]["characteristics"]["cpu"],
            architecture=flavor["spec"]["characteristics"]["architecture"],
            memory=flavor["spec"]["characteristics"]["memory"],
            gpu=flavor["spec"]["characteristics"]["gpu"],
            pods=flavor["spec"]["characteristics"]["pods"],
            ephemeral_storage=flavor["spec"]["characteristics"]["ephemeral-storage"],
            persistent_storage=flavor["spec"]["characteristics"]["persistent-storage"]
        ),
        owner=flavor["spec"]["owner"],
        optional_fields=flavor["spec"]["optionalFields"],
        policy=flavor["spec"]["policy"],
        price=flavor["spec"]["price"],
        location=flavor["spec"].get("location", {'latitude': -122.4194, 'longitude': 37.7749})
    )
