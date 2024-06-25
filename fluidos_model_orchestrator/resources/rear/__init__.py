from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any
from typing import cast

import kopf  # type: ignore
from kubernetes import client
from kubernetes.client.exceptions import ApiException

from fluidos_model_orchestrator.common import Flavor
from fluidos_model_orchestrator.common import Intent
from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.common import ResourceFinder
from fluidos_model_orchestrator.common import ResourceProvider
from fluidos_model_orchestrator.configuration import CONFIGURATION
from fluidos_model_orchestrator.configuration import Configuration


logger = logging.getLogger(__name__)


class REARResourceFinder(ResourceFinder):
    SOLVER_TIMEOUT = 25.0  # ~5 seconds
    SOLVER_SLEEPING_TIME = 0.2  # as float, in seconds ~200ms

    def __init__(self, configuration: Configuration = CONFIGURATION) -> None:
        self.api_client: client.CustomObjectsApi = client.CustomObjectsApi(api_client=configuration.k8s_client)
        self.identity: dict[str, str] = configuration.identity

    def find_best_match(self, request: Resource | Intent, namespace: str) -> list[ResourceProvider]:
        logger.info("Retrieving best match with REAR")

        if type(request) is Resource:
            resource = request
        elif type(request) is Intent:
            logger.info("Request is for \"intent\" resource")
        else:
            raise ValueError(f"Unkown resource type {type(request)}")

        logger.info("Request is for \"traditional\" resource")
        local = self._find_local(resource)

        # to be changed with something retrieved from the configuration
        if local is not None and (resource.region is None or resource.region.casefold() == "dublin"):
            logger.info(f"Found local resource {local=}")
            return [cast(ResourceProvider, local)]

        remote = self._find_remote(resource, namespace)

        if remote is not None:
            logger.info(f"Found remote resource {remote=}")
            return [remote]

        logger.info("No resource provider identified")
        return []

    def _initiate_search(self, body: dict[str, Any], namespace: str) -> str:
        logger.info("Initiating remote search")
        logger.debug(f"Solver body: {body}")

        try:
            response = self.api_client.get_namespaced_custom_object(
                group="nodecore.fluidos.eu",
                version="v1alpha1",
                namespace=namespace,
                plural="solvers",
                name=body["metadata"]["name"],
                async_req=False)
        except ApiException as e:
            logger.debug(f"Error retrieving {body['metadata']['name']}: {e=}")
            response = None

        if response is None or response["kind"] != "Solver":
            response = self.api_client.create_namespaced_custom_object(
                group="nodecore.fluidos.eu",
                version="v1alpha1",
                namespace=namespace,
                plural="solvers",
                body=body,
                async_req=False
            )
        else:
            logger.debug("Solver already existing")

        return response["metadata"]["name"]

    def _check_solver_status(self, solver_name: str, namespace: str) -> dict[str, Any] | None:
        logger.info(f"Retrieving solver/{solver_name} status")

        try:
            remote_flavour_status = self.api_client.get_namespaced_custom_object(
                group="nodecore.fluidos.eu",
                version="v1alpha1",
                namespace=namespace,
                plural="solvers",
                name=solver_name,
                async_req=False
            )
        except ApiException as e:
            logger.error("Unable to retrieve solver status")
            logger.debug(f"Reason: {e=}")
            return None

        logger.debug(f"Received {json.dumps(remote_flavour_status)}")

        return remote_flavour_status

    def _find_remote(self, resource: Resource, namespace: str) -> ResourceProvider | None:
        logger.info(f"Retrieving remote flavours in {namespace}")

        body, _ = self._resource_to_solver_request(resource, resource.id)

        kopf.adopt(body)

        solver_name = self._initiate_search(body, namespace)

        counter = 0

        while counter < self.SOLVER_TIMEOUT:
            time.sleep(self.SOLVER_SLEEPING_TIME)
            remote_flavour_status = self._check_solver_status(solver_name, namespace)

            if remote_flavour_status is None or "status" not in remote_flavour_status:
                return None

            phase: str = remote_flavour_status["status"]["solverPhase"]["phase"]

            if phase == "Solved":
                break

            if phase == "Failed" or phase == "Timed Out":
                logger.info("Unable to find matching flavour")
                return None

            if phase == "Running" or phase == "Pending":
                logger.debug("Still processing, wait")
                counter += 1
                continue
        else:
            logger.error("Solver did not finish withing the allocated time")
            return None

        # resource found and reserved, now we need to return the best matching
        peering_candidates = self._retrieve_peering_candidates(solver_name, namespace)
        if peering_candidates is None:
            logger.error("Error retrieving peering candidates from Discovery")
            return None

        if len(peering_candidates) == 0:
            logger.info("No valid peering candidates found")
            return None

        logger.debug(f"{peering_candidates=}")

        matching_resources = self._reserve_all(solver_name, peering_candidates, namespace)

        logger.debug(f"{matching_resources=}")

        return matching_resources[0]  # return first matching for now

    def _retrieve_peering_candidates(self, solver_name: str, namespace: str) -> list[dict[str, Any]] | None:
        logger.info(f"Retrieving discovery for {solver_name} in {namespace}")

        try:
            discovery = self.api_client.get_namespaced_custom_object(
                group="advertisement.fluidos.eu",
                version="v1alpha1",
                namespace=namespace,
                plural="discoveries",
                name=f"discovery-{solver_name}",
                async_req=False
            )

            return discovery.get("status", {}).get("peeringCandidateList", {}).get("items", None)

        except ApiException as e:
            logger.error("Unable to retrieve solver status")
            logger.debug(f"Reason: {e=}")
            return None

    def _reserve_all(self, solver_name: str, peering_candidates: list[dict[str, Any]], namespace: str) -> list[ResourceProvider]:
        logger.info("Reserving all peering candidates, just in case")
        return [
            candidate for candidate in
            [self._reserve_peering_candidate(solver_name, candidate, namespace) for candidate in peering_candidates]
            if candidate is not None
        ]

    def _create_reservation(self, solver_name: str, candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "apiVersion": "reservation.fluidos.eu/v1alpha1",
            "kind": "Reservation",
            "metadata": {
                "name": f'{candidate["metadata"]["name"]}-reservation'
            },
            "spec": {
                "solverID": solver_name,
                "buyer": self.identity,
                # Retrieve from PeeringCandidate Flavor Owner field
                "seller": candidate["spec"]["flavour"]["spec"]["owner"],
                # Set it to reserve
                "reserve": True,
                # Set it to purchase after reservation is completed and you have a transaction
                "purchase": False,
                # Retrieve from PeeringCandidate chosen to reserve
                "peeringCandidate": {
                    "name": candidate["metadata"]["name"],
                }
            }
        }

    def _reserve_peering_candidate(self, solver_name: str, candidate: dict[str, Any], namespace: str) -> RemoteResourceProvider | None:
        logger.info(f"Reserving peering candidate {candidate['metadata']['name']}")
        body = self._create_reservation(solver_name, candidate)

        kopf.adopt(body)

        try:
            response = self.api_client.create_namespaced_custom_object(
                group="reservation.fluidos.eu",
                version="v1alpha1",
                namespace=namespace,
                plural="reservations",
                body=body,
                async_req=False
            )
        except ApiException as e:
            logger.error(f"Unable to reserve {candidate['metadata']['name']}")
            logger.debug(f"Reason: {e=}")
            return None

        return RemoteResourceProvider(
            id=solver_name,
            peering_candidate=candidate["metadata"]["name"],
            local_cluster=self.identity,
            reservation=response["metadata"]["name"],
            namespace=namespace,
            api_client=self.api_client
        )

    def _resource_to_solver_request(self, resource: Resource, intent_id: str | None = None) -> tuple[dict[str, Any], str]:
        if intent_id is None:
            intent_id = str(uuid.uuid4())

        solver_request = {
            "apiVersion": "nodecore.fluidos.eu/v1alpha1",
            "kind": "Solver",
            "metadata": {
                "name": f"{intent_id}-solver"
            },
            "spec": {
                "intentID": intent_id,
                "findCandidate": True,
                "reserveAndBuy": False,
                "enstablishPeering": False,
                "selector": self._build_flavour_selector(resource)
            }
        }

        return (solver_request, intent_id)

    def _build_flavour_selector(self, resource: Resource) -> dict[str, Any]:
        return {
            "type": "k8s-fluidos",
            "architecture": resource.architecture if resource.architecture is not None else "amd64",
            "rangeSelector": self._build_range_selector(resource)
        }

    def _build_range_selector(self, resource: Resource) -> dict[str, str]:
        selector: dict[str, str] = {
            "minCpu": resource.cpu or "0n",
            "minMemory": resource.memory or "1Ki"
        }

        if resource.gpu is not None:
            selector["minGpu"] = resource.gpu

        return selector

    def _find_local(self, resource: Resource) -> list[LocalResourceProvider]:
        logger.info("Retrieving local flavours")

        local_flavours = self.api_client.list_namespaced_custom_object(
            group="nodecore.fluidos.eu",
            version="v1alpha1",
            plural="flavours",
            namespace="default"
        )

        if local_flavours is None:
            logger.info("Unable to retrieve flavours locally")
            return []

        if not len(local_flavours["items"]):
            logger.info("No flavours found locally")
            return []

        for flavour in local_flavours["items"]:
            name = flavour["metadata"]["name"]

            logger.info(f"Processing flavour {name=}")

            flavour_type = flavour["spec"]["type"]

            if flavour_type != "k8s-fluidos":
                logger.info(f"Skipping, wrong flavour type {flavour_type}")
                continue

            uid = flavour["metadata"]["uid"]

            architecture = flavour["spec"]["characteristics"]["architecture"]
            cpu = flavour["spec"]["characteristics"]["cpu"]
            gpu = flavour["spec"]["characteristics"]["gpu"]
            memory = flavour["spec"]["characteristics"]["memory"]

            if resource.can_run_on(Flavor(uid, cpu, architecture, gpu, memory)):
                logger.info("Local flavour is compatible, using it")
                return [LocalResourceProvider(uid)]

        logger.info("No suitable local flavour found")
        return []


class LocalResourceProvider(ResourceProvider):
    def __init__(self, id: str) -> None:
        super().__init__(id)

    def get_label(self) -> str:
        return ""


class RemoteResourceProvider(ResourceProvider):
    def __init__(self, id: str, local_cluster: Any, peering_candidate: str, reservation: str, namespace: str, api_client: client.CustomObjectsApi) -> None:
        super().__init__(id)
        self.local_cluster = local_cluster
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
