from __future__ import annotations

import logging
import uuid
from typing import Any

from fluidos_model_orchestrator.common.flavor import Flavor
from fluidos_model_orchestrator.common.flavor import FlavorCharacteristics
from fluidos_model_orchestrator.common.flavor import FlavorK8SliceData
from fluidos_model_orchestrator.common.flavor import FlavorMetadata
from fluidos_model_orchestrator.common.flavor import FlavorSpec
from fluidos_model_orchestrator.common.flavor import FlavorType
from fluidos_model_orchestrator.common.flavor import FlavorTypeData
from fluidos_model_orchestrator.common.intent import Intent
from fluidos_model_orchestrator.common.resource import ExternalResourceProvider
from fluidos_model_orchestrator.common.resource import Resource
from fluidos_model_orchestrator.common.resource import ResourceProvider
from fluidos_model_orchestrator.resources.rear.finder import REARResourceFinder


logger = logging.getLogger(__name__)


class ROBResourceFinder(REARResourceFinder):
    def find_best_match(self, resource: Resource, namespace: str, solver_name: str | None = None) -> list[ResourceProvider]:
        logger.info("Returning only local at the beginning, peering already in place")
        resources = self._find_local(resource=resource, namespace=namespace)

        logger.info("%d flavors found", len(resources))
        return resources

    def find_service(self, id: str, service: Intent, namespace: str) -> list[ExternalResourceProvider]:
        logger.info("Not supported")
        return []


_flavor = Flavor(
    metadata=FlavorMetadata(
        name=uuid.uuid4().hex,
        owner_references={}
    ),
    spec=FlavorSpec(
        availability=True,
        flavor_type=FlavorTypeData(
            type_data=FlavorK8SliceData(
                characteristics=FlavorCharacteristics(
                    cpu="",
                    architecture="",
                    memory=""
                )
            ),
            type_identifier=FlavorType.K8SLICE
        ),
        network_property_type="",
        owner={},
        providerID=""
    )
)


class DummyResourceProvider(ResourceProvider):
    def __init__(self, selector_key: str, selector_label: str) -> None:
        super().__init__(
            uuid.uuid4().hex,
            _flavor
        )
        self.selector_key = selector_key
        self.selector_label = selector_label

    def get_label(self) -> dict[str, str]:
        return {
            self.selector_key: self.selector_label
        }

    def acquire(self, namespace: str) -> bool:
        return True

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "DUMMY",
            "id": self.id,
            "key": self.selector_key,
            "label": self.selector_label
        }
