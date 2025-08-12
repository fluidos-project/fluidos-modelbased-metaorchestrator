from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from enum import unique
from typing import Any
from typing import cast

from fluidos_model_orchestrator.common.flavor import FlavorK8SliceData
from fluidos_model_orchestrator.common.flavor import FlavorType
from fluidos_model_orchestrator.common.resource import _check_gpu
from fluidos_model_orchestrator.common.resource import _cpu_compatible
from fluidos_model_orchestrator.common.resource import _memory_compatible
from fluidos_model_orchestrator.common.resource import ResourceProvider


logger = logging.getLogger(__name__)


_always_true: Callable[[ResourceProvider, str], bool] = lambda provider, value: True


def _validate_latency(value: str, data: list[Any]) -> bool:
    # assuming that it is falsified if the avg(last) X readings > required value
    if len(data) == 0:
        return True

    required_max_latency = float(value)
    values = data[0]["values"]

    avg_data = sum(int(value[1]) for value in values) / len(values)

    return avg_data <= required_max_latency


def _validate_throughput(value: str, data: dict[str, Any]) -> bool:
    return True


def _validate_battery_level(value: str, data: dict[str, Any]) -> bool:
    return False


def _monitor_bandwidth_against_point(value: str, data: dict[str, Any]) -> bool:
    return False


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


def _validate_sensor(provider: ResourceProvider, value: str) -> bool:
    if provider.flavor.spec.flavor_type.type_identifier is FlavorType.K8SLICE:
        properties = cast(FlavorK8SliceData, provider.flavor.spec.flavor_type.type_data).properties
        available_sensors: list[str] = properties.get("additionalProperties", {}).get("sensors", [])
        value = value.casefold()
        return any(
            value == v.casefold() for v in available_sensors
        )
    return False


def _validate_hardware(provider: ResourceProvider, value: str) -> bool:
    if provider.flavor.spec.flavor_type.type_identifier is FlavorType.K8SLICE:
        properties = cast(FlavorK8SliceData, provider.flavor.spec.flavor_type.type_data).properties
        additional_hardware: list[str] = properties.get("additionalProperties", {}).get("additional-hardware", [])
        value = value.casefold()
        return any(
            value == v.casefold() for v in additional_hardware
        )
    return False


def _validate_cyber_deception(provider: ResourceProvider, value: str) -> bool:
    if provider.flavor.spec.flavor_type.type_identifier is FlavorType.K8SLICE:
        properties = cast(FlavorK8SliceData, provider.flavor.spec.flavor_type.type_data).properties
        security_featues = properties.get("additionalProperties", {}).get("security_features", {})
        if "cyber_deception" in security_featues:
            return True
    return False


def _validate_magi(provider: ResourceProvider, value: str) -> bool:
    if provider.flavor.spec.flavor_type.type_identifier is FlavorType.K8SLICE:
        properties = cast(FlavorK8SliceData, provider.flavor.spec.flavor_type.type_data).properties
        security_featues = properties.get("additionalProperties", {}).get("security_features", {})
        if "magi" in security_featues:
            return True
    return False


def _validate_robot_status(provider: ResourceProvider, value: str) -> bool:
    # TO BE COMPLETED
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
    latency = "latency", False, _always_true, lambda args: f'fluidos_latency{{cluster="{args[0]}"}}[2m]', _validate_latency
    location = "location", False, validate_location
    throughput = "throughput", False, _always_true, lambda args: f'fluidos_throughput{{pod="{args[1]}/{args[2]}"}}[2m]', _validate_throughput
    compliance = "compliance", False, _validate_regulations
    energy = "energy", False, _always_true, True

    # ROB
    battery = "battery", False, _always_true, "fluidos_battery", _validate_battery_level
    robot_status = "robot-status", False, _validate_robot_status, lambda args: f'robot_status{{cluster="{args[0]}"}}', _always_true

    # carbon aware requests
    max_delay = "max-delay", False
    carbon_aware = "carbon-aware", False

    # TER
    bandwidth_against = "bandwidth-against", False, _validate_bandwidth_against_point, lambda args: f'fluidos_bandwidth_against{{cluster="{args[0]}"}}', _monitor_bandwidth_against_point
    tee_readiness = "tee-readiness", False, _validate_tee_available

    # service
    service = "service", True

    # mspl
    mspl = "mspl", False

    # sensors and hardware
    sensor = "sensor", False, _validate_sensor
    hardware = "hardware", False, _validate_hardware

    # security
    cyber_deception = "cyber-deception", False, _validate_cyber_deception
    magi = "magi", False, _validate_magi

    def __new__(cls, *args: str, **kwds: str) -> KnownIntent:
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, label: str, external: bool,
                 validator: Callable[[ResourceProvider, str], bool] = _always_true,
                 metric_name: Callable[[list[str]], str] | None = None,  # function getting remote_cluster_id, namespace, workload name
                 metric_validator: Callable[[str, Any], bool] | None = None):
        self.label = label
        self._external = external
        self._validator = validator
        self._metric_name = metric_name
        self._metric_validator = metric_validator

    def to_intent_key(self) -> str:
        return f"fluidos-intent-{self.label}"

    def is_external_requirement(self) -> bool:
        return self._external

    def validates(self, provider: ResourceProvider, value: str) -> bool:
        return self._validator(provider, value)

    def needs_monitoring(self) -> bool:
        return self._metric_name is not None

    def validate_monitoring(self, value: str, data: Any) -> bool:
        if self._metric_validator is None:
            raise ValueError("Trying to validate an intent not requiring monitoring")
        return self._metric_validator(value, data)

    def metric_name(self) -> Callable[[list[str]], str] | None:
        return self._metric_name

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
        return self.name.needs_monitoring()


def requires_validation(intent: Intent) -> bool:
    return intent.needs_monitoring()
