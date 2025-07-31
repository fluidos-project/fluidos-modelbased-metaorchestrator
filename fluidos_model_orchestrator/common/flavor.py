from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from enum import auto
from enum import Enum
from enum import unique
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class GPUData:
    cores: int | str = 0
    memory: int | str = 0
    model: str = ""

    def can_run_on(self, other: GPUData) -> bool:
        return int(other.cores) >= int(self.cores) and int(other.memory) >= int(self.memory)

    def to_json(self) -> dict[str, Any]:
        return {
            "cores": self.cores,
            "memory": self.memory,
            "model": self.model,
        }


@dataclass(kw_only=True)
class FlavorCharacteristics:
    cpu: str
    architecture: str
    memory: str
    gpu: GPUData | str | int = "0"
    pods: str | None = None
    storage: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "cpu": self.cpu,
            "gpu": self.gpu.to_json() if type(self.gpu) is GPUData else self.gpu,
            "memory": self.memory,
            "pods": self.pods,
            "storage": self.storage,
        }


@dataclass(kw_only=True)
class FlavorK8SliceData:
    characteristics: FlavorCharacteristics
    policies: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "characteristics": self.characteristics.to_json(),
            "policies": self.policies,
            "properties": self.properties,
        }


@dataclass(kw_only=True)
class FlavorServiceData:
    category: str
    configurationTemplate: dict[str, Any]
    description: str
    hostingPolicies: list[str]
    name: str
    tags: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "configurationTemplate": self.configurationTemplate,
            "description": self.description,
            "hostingPolicies": self.hostingPolicies,
            "name": self.name,
            "tags": self.tags,
        }


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
        if type_name == "Service":
            return FlavorType.SERVICE

        raise ValueError(f"Not supported {type_name=}")

    def to_json(self) -> str:
        if self is FlavorType.K8SLICE:
            return "K8Slice"
        if self is FlavorType.SERVICE:
            return "Service"

        raise ValueError("Not sure what to do")


@dataclass(kw_only=True)
class FlavorTypeData:
    type_identifier: FlavorType
    type_data: FlavorK8SliceData | FlavorServiceData

    def to_json(self) -> dict[str, Any]:
        return {
            "typeIdentifier": self.type_identifier,
            "typeData": self.type_data.to_json(),
        }


@dataclass(kw_only=True)
class FlavorSpec:
    availability: bool
    flavor_type: FlavorTypeData
    network_property_type: str
    owner: dict[str, Any]
    providerID: str
    price: dict[str, Any] = field(default_factory=dict)
    location: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "availability": self.availability,
            "flavorType": self.flavor_type.to_json(),
            "networkPropertyType": self.network_property_type,
            "owner": self.owner,
            "providerID": self.providerID,
            "price": self.price,
            "location": self.location,
        }


@dataclass(kw_only=True)
class FlavorMetadata:
    name: str
    owner_references: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ownerReferences": self.owner_references
        }


@dataclass(kw_only=True)
class Flavor:
    metadata: FlavorMetadata
    spec: FlavorSpec

    def to_json(self) -> dict[str, Any]:
        return {
            "apiVersion": "nodecore.fluidos.eu/v1alpha1",
            "kind": "Flavor",
            "metadata": self.metadata.to_json(),
            "spec": self.spec.to_json(),
        }


def build_flavor(flavor: dict[str, Any]) -> Flavor:
    if "kind" in flavor:
        if flavor["kind"] != "Flavor":
            raise ValueError(f"Unable to process kind {flavor['kind']}")
    else:
        logger.info("Building flavor from spec, not object")

    return Flavor(
        metadata=_build_metadata(flavor["metadata"]),
        spec=_build_spec(flavor["spec"]),
    )


def _build_metadata(metadata: dict[str, Any]) -> FlavorMetadata:
    return FlavorMetadata(
        name=metadata["name"],
        owner_references=metadata.get("ownerReferences", {}),
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


def _build_flavor_type_data(flavor_type: FlavorType, data: dict[str, Any]) -> FlavorK8SliceData | FlavorServiceData:
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
    elif flavor_type is FlavorType.SERVICE:
        return FlavorServiceData(
            category=data["category"],
            configurationTemplate=data["configurationTemplate"],
            description=data["description"],
            hostingPolicies=data["hostingPolicies"],
            name=data["name"],
            tags=data["tags"],
        )

    raise ValueError(f"Unsupported flavor type: {flavor_type}")
