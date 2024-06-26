from __future__ import annotations

import logging
import uuid
from typing import Any

import kopf  # type: ignore
from kubernetes import client
from kubernetes.client.exceptions import ApiException

from fluidos_model_orchestrator.common import Flavor
from fluidos_model_orchestrator.common import ResourceProvider

logger = logging.getLogger(__name__)


class RemoteResourceProvider(ResourceProvider):
    def __init__(self, id: str, flavor: Flavor, peering_candidate: str, reservation: str, namespace: str, api_client: client.CustomObjectsApi) -> None:
        super().__init__(id, flavor)
        self.peering_candidate = peering_candidate
        self.reservation = reservation
        self.namespace = namespace
        self.api_client = api_client

    def acquire(self) -> bool:
        logger.info("Creating connection to remote node")
        contract = self._buy()
        if contract is None:
            return False
        return self._establish_peering(contract)

    def get_label(self) -> str:
        raise NotImplementedError()

    def _buy(self) -> dict[str, Any] | None:
        logger.info(f"Establishing buying of {self.peering_candidate}")

        try:
            reservation = self.api_client.patch_namespaced_custom_object(
                group="reservation.fluidos.eu",
                version="v1alpha1",
                namespace=self.namespace,
                plural="reservations",
                name=self.reservation,
                body={
                    "spec": {
                        "purchase": True
                    }
                },
                async_req=False)

            logger.debug(f"Retrieved {reservation=}")

            contract_name = reservation["status"]["contract"]["name"]

            logger.debug(f"Retrieving contract {contract_name}")

            # retrieve contract ID or fail
            contract = self.api_client.get_namespaced_custom_object(
                group="reservation.fluidos.eu",
                version="v1alpha1",
                namespace=self.namespace,
                plural="contracts",
                name=contract_name,
            )

            logger.debug(f"Retrieved {contract=}")

            return contract
        except ApiException as e:
            logger.error(f"Error buying {self.reservation}")
            logger.debug(f"{e=}")

        return None

    def _establish_peering(self, contract: dict[str, Any]) -> bool:
        logger.info(f"Establishing peering for {self.peering_candidate}")

        allocation_name = f"{self.id}-allocation"

        body = {
            "kind": "Allocation",
            "metadata": {
                "name": allocation_name
            },
            "spec": {
                # From the reservation get the contract and from the contract get the Spec.SellerCredentials.ClusterID
                "remoteClusterID": contract["sellerCredentials"]["clusterID"],
                # Get it from the solver
                "intentID": self.id,
                # Set a name for the VirtualNode on the consumer cluster. Pattern suggested: "liqo-clusterName", where clusterName s the one you get from the contract.Spec.SellerCredentials.ClusterName
                "nodeName": f"liqo-{str(uuid.uuid4())}",
                # On the consumer set it as VirtualNode, since the allocation will be bound to a VirtualNode to be created
                "type": "VirtualNode",
                # On the consumer set it as Local, since the allocation of resources will be consumed locally
                "destination": "Local",
                # Retrieve information from the reservation and the contract bou d to it
                "contract": {
                    "name": contract["metadata"]["name"],
                    "namespace": contract["metadata"]["namespace"],
                }
            }
        }

        kopf.adopt(body)

        try:
            self.api_client.create_namespaced_custom_object(
                group="nodecore.fluidos.eu",
                version="v1alpha1",
                namespace=self.namespace,
                plural="allocations",
                name=allocation_name,
                body=body,
                async_req=False)

        except ApiException as e:
            logger.error(f"Error establishing peering for {self.peering_candidate}")
            logger.debug(f"{e=}")

        return False
