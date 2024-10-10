import logging
from typing import Any

from kubernetes import client  # type: ignore

from fluidos_model_orchestrator.common import ServiceResourceProvider
from fluidos_model_orchestrator.flavor import Flavor
from fluidos_model_orchestrator.resources.rear.remote_resource_provider import RemoteResourceProvider


logger = logging.getLogger(__name__)


class REARServiceResourceProvider(ServiceResourceProvider, RemoteResourceProvider):
    def __init__(self, id: str, flavor: Flavor, peering_candidate: str, reservation: str, namespace: str, api_client: client.CustomObjectsApi, seller: dict[str, Any]) -> None:
        super().__init__(id, flavor)
        self.peering_candidate = peering_candidate
        self.reservation = reservation
        self.namespace = namespace
        self.api_client = api_client
        self.seller = seller
        self.contract: str | None = None

    def acquire(self) -> bool:
        logger.info("Creating connection to remote node")
        contract = self._buy()
        if contract is None:
            return False
        self.contract = contract
        return self._establish_peering(contract)
