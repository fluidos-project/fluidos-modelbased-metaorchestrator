from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import kopf  # type: ignore
from kubernetes import client  # type: ignore
from kubernetes.client.exceptions import ApiException  # type: ignore

from fluidos_model_orchestrator.common import build_flavor
from fluidos_model_orchestrator.common import Flavor
from fluidos_model_orchestrator.common import FlavorType
from fluidos_model_orchestrator.common import Intent
from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.common import ResourceFinder
from fluidos_model_orchestrator.common import ResourceProvider
from fluidos_model_orchestrator.configuration import CONFIGURATION
from fluidos_model_orchestrator.configuration import Configuration
from fluidos_model_orchestrator.resources.rear.local_resource_provider import LocalResourceProvider
from fluidos_model_orchestrator.resources.rear.remote_resource_provider import RemoteResourceProvider

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
            return []  # not supported yet
        else:
            raise ValueError(f"Unkown resource type {type(request)}")

        logger.info("Request is for \"traditional\" resource")
        local: list[ResourceProvider] = self._find_local(resource, namespace)

        if len(local):
            logger.info(f"Found local resource {local=}")

        remote = self._find_remote(resource, namespace)

        logger.info(f"Found remote resource {remote=}")

        return local + remote

    def retrieve_all_flavors(self, namespace: str) -> list[Flavor]:
        logger.info("Retrieving all flavours")

        locally_available_flavours = self._get_locally_available_flavors(namespace)
        logger.debug(f"Retrieved {len(locally_available_flavours)} local flavors")

        remotely_available_flavours = self._get_remotely_available_flavors(namespace)
        logger.debug(f"Retrieved {len(remotely_available_flavours)} remote flavors")

        return locally_available_flavours + remotely_available_flavours

    def update_local_flavor(self, flavor: Flavor, data: Any, namespace: str) -> None:
        logger.info(f"Updating {flavor=} with {data=}")

        response = self.api_client.patch_namespaced_custom_object(
            group="nodecore.fluidos.eu",
            version="v1alpha1",
            namespace=namespace,
            plural="flavors",
            name=flavor.metadata.name,
            body={
                "spec": {
                    "flavorType": {
                        "typeData": {
                            "properties": data
                        }
                    }
                }
            }
        )

        if response is None:
            raise ValueError()

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

    def _find_remote(self, resource: Resource, namespace: str) -> list[ResourceProvider]:
        logger.info(f"Retrieving remote flavours in {namespace}")

        body, _ = self._resource_to_solver_request(resource, resource.id)

        kopf.adopt(body)

        solver_name = self._initiate_search(body, namespace)

        counter = 0

        while counter < self.SOLVER_TIMEOUT:
            time.sleep(self.SOLVER_SLEEPING_TIME)
            remote_flavour_status = self._check_solver_status(solver_name, namespace)

            if remote_flavour_status is None or "status" not in remote_flavour_status:
                return []

            phase: str = remote_flavour_status["status"]["solverPhase"]["phase"]

            if phase == "Solved":
                break

            if phase == "Failed" or phase == "Timed Out":
                logger.info("Unable to find matching flavour")
                return []

            if phase == "Running" or phase == "Pending":
                logger.debug("Still processing, wait")
                counter += 1
                continue
        else:
            logger.error("Solver did not finish withing the allocated time")
            return []

        # resource found and reserved, now we need to return the best matching
        peering_candidates = self._retrieve_peering_candidates(solver_name, namespace)
        if peering_candidates is None:
            logger.error("Error retrieving peering candidates from Discovery")
            return []

        if len(peering_candidates) == 0:
            logger.info("No valid peering candidates found")
            return []

        logger.debug(f"{peering_candidates=}")

        matching_resources: list[ResourceProvider] = self._reserve_all(solver_name, peering_candidates, namespace)

        logger.debug(f"{matching_resources=}")

        return matching_resources

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

            logger.debug(f"{response=}")
        except ApiException as e:
            logger.error(f"Unable to reserve {candidate['metadata']['name']}")
            logger.debug(f"Reason: {e=}")
            return None

        return RemoteResourceProvider(
            id=solver_name,
            flavor=build_flavor(candidate["spec"]["flavour"]),
            peering_candidate=candidate["metadata"]["name"],
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

    def _find_local(self, resource: Resource, namespace: str) -> list[ResourceProvider]:
        logger.info("Retrieving locally available flavours")

        fitting_resources: list[ResourceProvider] = []

        local_flavours = self._get_locally_available_flavors(namespace)

        for flavor in local_flavours:
            name = flavor.metadata.name

            logger.info(f"Processing flavor {name=}")

            if flavor.spec.flavor_type.type_identifier is not FlavorType.K8SLICE:
                logger.info(f"Skipping, wrong flavour type {flavor.spec.flavor_type}")
                continue

            if resource.can_run_on(flavor):
                logger.info("Local flavour is compatible, using it")
                fitting_resources.append(
                    LocalResourceProvider(
                        flavor.metadata.name,
                        flavor
                    ))

        return fitting_resources

    def _get_locally_available_flavors(self, namespace: str) -> list[Flavor]:
        flavor_list: dict[str, Any]

        try:
            flavor_list = self.api_client.list_namespaced_custom_object(
                group="nodecore.fluidos.eu",
                version="v1alpha1",
                plural="flavors",
                namespace=namespace,
            )
        except ApiException:
            logger.warn("Failed to retrieve local flavors, is node available?")
            flavor_list = {}

        return [build_flavor(flavor) for flavor in flavor_list.get("items", [])]

    def _get_remotely_available_flavors(self, namespace: str) -> list[Flavor]:
        return []  # TODO: waiting for REAR 2
