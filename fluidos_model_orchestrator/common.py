from __future__ import annotations

import json
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
    storage: str | None = None
    region: str | None = None
    pods: str | int | None = None

    def can_run_on(self, flavor: Flavor) -> bool:
        logger.debug(f"Testing {self=} against {flavor=}")
        if self.cpu is not None and not _cpu_compatible(self.cpu, flavor.spec.flavor_type.type_data.characteristics.cpu):
            return False

        if self.memory is not None and not _memory_compatible(self.memory, flavor.spec.flavor_type.type_data.characteristics.memory):
            return False

        if self.architecture is not None and self.architecture != flavor.spec.flavor_type.type_data.characteristics.architecture:
            return False

        if self.gpu is not None:
            our_gpu = _convert_to_gpudata(self.gpu)
            their_gpu = _convert_to_gpudata(flavor.spec.flavor_type.type_data.characteristics.gpu)

            return our_gpu.can_run_on(their_gpu)

        if self.pods is not None:
            return int(self.pods) == int(flavor.spec.flavor_type.type_data.characteristics.pods if flavor.spec.flavor_type.type_data.characteristics.pods is not None else 0)

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
class GPUData:
    cores: int | str = 0
    memory: int | str = 0
    model: str = ""

    def can_run_on(self, other: GPUData) -> bool:
        return int(other.cores) >= int(self.cores) and int(other.memory) >= int(self.memory)


@dataclass(kw_only=True)
class FlavorCharacteristics:
    cpu: str
    architecture: str
    memory: str
    gpu: GPUData | str | int = "0"
    pods: str | None = None
    storage: str | None = None


@unique
class FlavorType(Enum):
    K8SLICE = auto()
    VM = auto()
    SERVICE = auto()
    SENSOR = auto()

    @staticmethod
    def factory(type_name: str) -> FlavorType:
        if type_name == "K8Slice":
            return FlavorType.K8SLICE

        raise ValueError(f"Not supported {type_name=}")


@dataclass(kw_only=True)
class FlavorMetadata:
    name: str
    owner_references: dict[str, Any]


@dataclass(kw_only=True)
class FlavorK8SliceData:
    characteristics: FlavorCharacteristics
    policies: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class FlavorTypeData:
    type_identifier: FlavorType
    type_data: FlavorK8SliceData


@dataclass(kw_only=True)
class FlavorSpec:
    availability: bool
    flavor_type: FlavorTypeData
    network_property_type: str
    owner: dict[str, Any]
    providerID: str
    price: dict[str, Any] = field(default_factory=dict)
    location: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Flavor:
    metadata: FlavorMetadata
    spec: FlavorSpec


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
    for field_name, field_value in provider.flavor.spec.flavor_type.type_data.properties.items():
        if str(field_value).casefold() == value:
            return True

    return False


def _check_cpu(provider: ResourceProvider, value: str) -> bool:
    return _cpu_compatible(value, provider.flavor.spec.flavor_type.type_data.characteristics.cpu)


def _check_memory(provider: ResourceProvider, value: str) -> bool:
    return _memory_compatible(value, provider.flavor.spec.flavor_type.type_data.characteristics.memory)


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
    data = json.loads(value)

    gpu: GPUData = _convert_to_gpudata(provider.flavor.spec.flavor_type.type_data.characteristics.gpu)

    if type(data) is dict:
        return gpu.can_run_on(_convert_to_gpudata(data))
    elif type(data) is int:
        return int(gpu.cores) >= data

    return False


def validate_location(provider: ResourceProvider, value: str) -> bool:
    value = value.casefold()

    location = provider.flavor.spec.location

    return any(
        value == str(_val).casefold() for _val in location.values()
    )


@unique
class KnownIntent(Enum):
    # k8s resources
    cpu = "cpu", False, _check_cpu
    memory = "memory", False, _check_memory
    gpu = "gpu", False, _check_gpu
    architecture = "architecture", False, lambda provider, value: value == provider.flavor.spec.flavor_type.type_data.characteristics.architecture

    # high order requests
    latency = "latency", False, _always_true
    location = "location", False, validate_location
    throughput = "throughput", False, _always_true
    compliance = "compliance", False, _validate_regulations
    energy = "energy", False, _always_true
    battery = "battery", False, _always_true

    # carbon aware requests
    max_delay = "max_delay", False, _always_true
    carbon_aware = "carbon_aware", False, _always_true

    #mspl
    #mspl = "mspl", False, _always_true

    # service
    service = "service", True, _always_true

    def __new__(cls, *args: str, **kwds: str) -> KnownIntent:
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, external: bool, validator: Callable[[ResourceProvider, str], bool]):
        self._external = external
        self._validator = validator

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
    #print(f"flavor!!!!!!!!!!!!!!!!!!!!!{flavor}!!!!!!!!!!!!!")
    
    if flavor["kind"] != "Flavor":
        raise ValueError(f"Unable to process kind {flavor['kind']}")

    return Flavor(
        metadata=_build_metadata(flavor["metadata"]),
        spec=_build_spec(flavor["spec"]),
    )


def _build_metadata(metadata: dict[str, Any]) -> FlavorMetadata:
    return FlavorMetadata(
        name=metadata["name"],
        owner_references=metadata["ownerReferences"],
    )


def _build_spec(spec: dict[str, Any]) -> FlavorSpec:
    return FlavorSpec(
        availability=spec["availability"],
        flavor_type=_build_flavor_type(spec["flavorType"]),
        location=spec["location"],
        network_property_type=spec["networkPropertyType"],
        owner=spec["owner"],
        price=spec["price"],
        providerID=spec["providerID"],
    )


def _build_flavor_type(flavor_type_data: dict[str, Any]) -> FlavorTypeData:
    flavor_type = FlavorType.factory(flavor_type_data["typeIdentifier"])
    return FlavorTypeData(
        type_identifier=flavor_type,
        type_data=_build_flavor_type_data(flavor_type, flavor_type_data["typeData"])
    )


def _build_flavor_type_data(flavor_type: FlavorType, data: dict[str, Any]) -> FlavorK8SliceData:
    if flavor_type is FlavorType.K8SLICE:
        return FlavorK8SliceData(
            characteristics=FlavorCharacteristics(
                cpu=data["characteristics"]["cpu"],
                architecture=data["characteristics"]["architecture"],
                memory=data["characteristics"]["memory"],
                gpu=data["characteristics"]["gpu"],
                pods=data["characteristics"]["pods"],
                storage=data["characteristics"]["storage"],
            ),
            policies=data.get("policies", {}),
            properties=data.get("properties", {})
        )
    raise ValueError(f"Unsupported flavor type: {flavor_type}")
