from __future__ import annotations
from abc import ABC, abstractmethod
import json
import time
from typing import Any, Optional
import uuid

import kopf

from ..common import Flavour, Intent
from ..common import Resource
from ..common import ModelPredictRequest
from ..common import ModelPredictResponse

from kubernetes import config, client


import logging


logger = logging.getLogger(__name__)


class ResourceProvider(ABC):
    def __init__(self, id: str) -> None:
        self.id = id

    def reserve(self) -> bool:
        return True

    def acquire(self) -> bool:
        return True

    @abstractmethod
    def get_label(self) -> str:
        raise NotImplementedError("Abstract method")


class LocalResourceProvider(ResourceProvider):
    def __init__(self, id: str) -> None:
        super().__init__(id)

    def get_label(self) -> str:
        return ""


class RemoteResourceProvider(ResourceProvider):
    def __init__(self, id: str, local_cluster: str, remote_cluster: str, peering_candidate: str) -> None:
        super().__init__(id)
        self.local_cluster = local_cluster
        self.remote_cluster = remote_cluster
        self.peering_candidate = peering_candidate

    def reserve(self) -> bool:
        logger.info("Reserving remote resource")
        time.sleep(0.3)

        return True

    def acquire(self) -> bool:
        logger.info("Creating connection to remote node")

        return True

    # def acquire(self) -> bool:
    #     logger.info("Creating connection remote node")

    #     try:
    #         response = subprocess.run(f"liqoctl generate peer-command --kubeconfig \"$PWD/utils/examples/{self.remote_cluster}-kubeconfig.yaml\"", shell=True, check=True, capture_output=True)
    #     except subprocess.CalledProcessError as e:
    #         logger.debug("BEGIN -------")
    #         logger.debug(e)
    #         logger.debug("-------")
    #         logger.debug(e.stderr)
    #         logger.debug("-------")
    #         logger.debug(e.stdout)
    #         logger.debug("END -------")

    #     peering_command = _extract_command(response.stdout)

    #     self.remote_cluster_id = _extract_remote_cluster_id(peering_command)

    #     subprocess.run(peering_command, shell=True, check=True, capture_output=True)  # because it should be running on the acquiring cluster

    #     subprocess.run(f"kubectl delete solver/{self.id}", shell=True, check=False, capture_output=True)

    #     if self.peering_candidate is not None:
    #         subprocess.run(f"kubectl delete peeringcandidate/{self.peering_candidate}", shell=True, check=False, capture_output=True)

    #     return True

    def get_label(self) -> str:
        return self.remote_cluster_id


# def _extract_remote_cluster_id(peering_command: str) -> str:
#     components = peering_command.split()
#     for id, token in enumerate(components):
#         if token == "--cluster-id":  # nosec
#             return components[id + 1]
#     raise ValueError("Missing cluster-id")


# def _extract_command(message: bytes) -> str:
#     logger.debug(f"Received {message=}")
#     text = str(message, encoding="utf-8")

#     return text.strip().split("\n")[-1]


class ResourceFinder(ABC):
    def find_best_match(self, resource: Resource | Intent) -> ResourceProvider | None:
        raise NotImplementedError()


def get_resource_finder(request: ModelPredictRequest, predict: ModelPredictResponse) -> ResourceFinder:
    logger.info("REARResourceFinder being returned")
    return REARResourceFinder()


class REARResourceFinder(ResourceFinder):
    def __init__(self) -> None:
        config.load_config()
        self.api_client = client.CustomObjectsApi()

    def find_best_match(self, request: Resource | Intent) -> ResourceProvider | None:
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
            return local

        remote = self._find_remote(resource)

        if remote is not None:
            logger.info(f"Found remote resource {remote=}")
            return remote
        logger.info("No resource provider identified")
        return None

    def _initiate_search(self, body: dict[str, Any]) -> str:
        logger.info("Initiating remote search")

        response = self.api_client.create_namespaced_custom_object(
            group="nodecore.fluidos.eu",
            version="v1alpha1",
            namespace="default",
            plural="solvers",
            body=body,
            async_req=False
        )

        logger.debug(f"Response: {response}")

        return response["metadata"]["name"]

    def _check_solver_status(self, solver_name: str) -> dict[str, Any]:
        logger.info(f"Retrieving solver {solver_name} status")

        remote_flavour_status = self.api_client.get_namespaced_custom_object(
            group="nodecore.fluidos.eu",
            version="v1alpha1",
            namespace="default",
            plural="solvers",
            name=solver_name,
            async_req=False
        )

        logger.debug(f"Received {json.dumps(remote_flavour_status)}")

        return remote_flavour_status

    def _find_remote(self, resource: Resource) -> ResourceProvider | None:
        logger.info("Retrieving remote flavours")

        body, _ = self._resource_to_rear_request(resource, resource.id)

        kopf.adopt(body)

        solver_name = self._initiate_search(body)

        while True:
            time.sleep(0.2)
            remote_flavour_status = self._check_solver_status(solver_name)

            if "status" not in remote_flavour_status:
                logger.info("Solver request not yet handled, wait")
                continue

            phase = remote_flavour_status["status"]["solverPhase"]["phase"]

            if phase == "Solved":
                return RemoteResourceProvider(id=solver_name, peering_candidate=remote_flavour_status["status"]["peeringCandidate"]["name"], remote_cluster="milan", local_cluster="dublin")

            if phase == "Failed" or phase == "Timed Out":
                logger.info("Unable to find matching flavour")
                return None

            if phase == "Running" or phase == "Pending":
                logger.info("Still processing, wait")

    def _resource_to_rear_request(self, resource: Resource, intent_id: Optional[str] = None) -> tuple[dict[str, Any], str]:
        if intent_id is None:
            intent_id = str(uuid.uuid4())

        solver_request = {
            "apiVersion": "nodecore.fluidos.eu/v1alpha1",
            "kind": "Solver",
            "metadata": {
                "name": intent_id
            },
            "spec": {
                "intentID": intent_id,
                "findCandidate": True,
                "selector": self._build_flavour_selector(resource)
            }
        }

        return (solver_request, intent_id)

    def _build_flavour_selector(self, resource: Resource) -> dict[str, Any]:
        return {
            "type": "k8s-fluidos",
            "architecture": resource.architecture if resource.architecture is not None else "x86_64",
            "rangeSelector": self._build_range_selector(resource)
        }

    def _build_range_selector(self, resource: Resource) -> dict[str, str]:
        selector: dict[str, str] = {
            "minCpu": resource.cpu or "0n",
            "minMemory": resource.memory or "1KiB"
        }

        if resource.gpu is not None:
            selector["minGpu"] = resource.gpu

        return selector

    def _find_local(self, resource: Resource) -> LocalResourceProvider | None:
        logger.info("Retrieving local flavours")

        local_flavours = self.api_client.list_namespaced_custom_object(
            group="nodecore.fluidos.eu",
            version="v1alpha1",
            plural="flavours",
            namespace="default"
        )

        if local_flavours is None:
            logger.info("Unable to retrieve flavours locally")
            return None

        if not len(local_flavours["items"]):
            logger.info("No flavours found locally")
            return None

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

            if resource.can_run_on(Flavour(uid, cpu, architecture, gpu, memory)):
                logger.info("Local flavour is compatible, using it")
                return LocalResourceProvider(uid)

        logger.info("No suitable local flavour found")
        return None
