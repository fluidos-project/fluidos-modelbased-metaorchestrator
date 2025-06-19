from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from enum import unique
from typing import cast

from fluidos_model_orchestrator.common.flavor import FlavorK8SliceData
from fluidos_model_orchestrator.common.flavor import FlavorType
from fluidos_model_orchestrator.common.resource import _check_gpu
from fluidos_model_orchestrator.common.resource import _cpu_compatible
from fluidos_model_orchestrator.common.resource import _memory_compatible
from fluidos_model_orchestrator.common.resource import ResourceProvider


logger = logging.getLogger(__name__)


_always_true: Callable[[ResourceProvider, str], bool] = lambda provider, value: True


def _validate_bandwidth_against_point(provider: ResourceProvider, value: str) -> bool:
    if provider.flavor.spec.flavor_type.type_identifier is not FlavorType.K8SLICE:
        return False

    # assumes value is of the form "<operator> value <point>"
    [operator, quantity, point] = value.split(" ")

    type_data = cast(FlavorK8SliceData, provider.flavor.spec.flavor_type.type_data)

    bandwidth_properties = type_data.properties.get("additionalProperties", {}).get("bandwidth", {}).get(point, None)

    if bandwidth_properties is not None:
        # assume that bandwidth_property is in ms
        bandwidth = int(bandwidth_properties[:-2])
        required_bandwidth = int(quantity[:-2])   # assumes quantity = "\d+ms"

        match operator:
            case ">":
                return bandwidth > required_bandwidth
            case ">=":
                return bandwidth >= required_bandwidth
            case "<":
                return bandwidth < required_bandwidth
            case "<=":
                return bandwidth <= required_bandwidth
            case "=":
                return bandwidth == required_bandwidth
            case _:
                raise ValueError(f"Unknown operator {operator=}")
    return False


def _validate_architecture(provider: ResourceProvider, value: str) -> bool:
    if provider.flavor.spec.flavor_type.type_identifier is FlavorType.K8SLICE:
        return value == cast(FlavorK8SliceData, provider.flavor.spec.flavor_type.type_data).characteristics.architecture
    return False


def _validate_tee_available(provider: ResourceProvider, value: str) -> bool:
    if provider.flavor.spec.flavor_type.type_identifier is FlavorType.K8SLICE:
        properties = cast(FlavorK8SliceData, provider.flavor.spec.flavor_type.type_data).properties
        return value.capitalize() == str(properties.get("additionalProperties", {}).get("TEE", "False")).capitalize()
    return False


def validate_location(provider: ResourceProvider, value: str) -> bool:
    value = value.casefold()

    location = provider.flavor.spec.location

    for val in location.values():
        val = str(val).casefold()

        if val == value:
            logger.debug("Returning True")
            return True

    logger.debug("Returning False")
    return False


def _validate_regulations(provider: ResourceProvider, value: str) -> bool:
    """
    Assumes values of the form:
    GDPR
    DORA
    or such
    """
    value = value.casefold()
    if provider.flavor.spec.flavor_type.type_identifier is FlavorType.K8SLICE:
        for _, field_value in cast(FlavorK8SliceData, provider.flavor.spec.flavor_type.type_data).properties.items():
            if str(field_value).casefold() == value:
                return True

    return False


def _check_cpu(provider: ResourceProvider, value: str) -> bool:
    if provider.flavor.spec.flavor_type.type_identifier is FlavorType.K8SLICE:
        return _cpu_compatible(value, cast(FlavorK8SliceData, provider.flavor.spec.flavor_type.type_data).characteristics.cpu)
    return False


def _check_memory(provider: ResourceProvider, value: str) -> bool:
    if provider.flavor.spec.flavor_type.type_identifier is FlavorType.K8SLICE:
        return _memory_compatible(value, cast(FlavorK8SliceData, provider.flavor.spec.flavor_type.type_data).characteristics.memory)
    return False


def _validate_vm_type(provider: ResourceProvider, value: str) -> bool:
    if provider.flavor.spec.flavor_type.type_identifier is FlavorType.K8SLICE:
        return cast(FlavorK8SliceData, provider.flavor.spec.flavor_type.type_data).properties.get("additionalProperties", {}).get("vm-type", "") == value

    return False


@unique
class KnownIntent(Enum):
    # k8s resources
    cpu = "cpu", False, _check_cpu
    memory = "memory", False, _check_memory
    gpu = "gpu", False, _check_gpu
    architecture = "architecture", False, _validate_architecture

    # Node VM charactecteristics
    vm_type = "vm-type", False, _validate_vm_type

    # high order requests
    latency = "latency", False, _always_true, True
    location = "location", False, validate_location
    throughput = "throughput", False, _always_true, True
    compliance = "compliance", False, _validate_regulations
    energy = "energy", False, _always_true, True
    battery = "battery", False, _always_true, True

    # carbon aware requests
    max_delay = "max-delay", False, _always_true
    carbon_aware = "carbon-aware", False, _always_true

    # TER
    bandwidth_against = "bandwidth-against", False, _validate_bandwidth_against_point, True
    tee_readiness = "tee-readiness", False, _validate_tee_available

    # service
    service = "service", True, _always_true

    #mspl
    mspl = "mspl", False, _always_true

    def __new__(cls, *args: str, **kwds: str) -> KnownIntent:
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, label: str, external: bool, validator: Callable[[ResourceProvider, str], bool], needs_monitoring: bool = False):
        self.label = label
        self._external = external
        self._validator = validator
        self._needs_monitoring = needs_monitoring

    def to_intent_key(self) -> str:
        return f"fluidos-intent-{self.label}"

    def is_external_requirement(self) -> bool:
        return self._external

    def validates(self, provider: ResourceProvider, value: str) -> bool:
        return self._validator(provider, value)

    @staticmethod
    def is_supported(intent_name: str) -> bool:
        if intent_name.startswith("fluidos-intent-"):
            intent_name = "-".join(intent_name.split("-")[2:])

        return any(
            known_intent.label == intent_name for known_intent in KnownIntent
        )

    @staticmethod
    def get_intent(intent_name: str) -> KnownIntent:
        # defensive programming
        if not KnownIntent.is_supported(intent_name):
            raise ValueError(f"Unsupported intent: {intent_name=}")

        name = "-".join(intent_name.split("-")[2:]).casefold()
        logger.info(f"Received intent: {name}")
        return next(known_intent for known_intent in KnownIntent if known_intent.label == name)


@dataclass
class Intent:
    name: KnownIntent
    value: str

    def is_external_requirement(self) -> bool:
        return self.name.is_external_requirement()

    def validates(self, provider: ResourceProvider) -> bool:
        return self.name.validates(provider, self.value)

    def needs_monitoring(self) -> bool:
        return self.name._needs_monitoring


def requires_validation(intent: Intent) -> bool:
    return intent.needs_monitoring()


def has_intent_validation_failed(intent: Intent, prometheus_ref: str) -> bool:
    return False
