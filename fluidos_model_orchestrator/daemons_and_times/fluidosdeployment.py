import asyncio
import datetime
from logging import Logger
from typing import Any
from uuid import uuid4

import kopf  # type: ignore
from kopf._cogs.structs import bodies  # type: ignore
from kopf._cogs.structs import patches  # type: ignore
from kubernetes.client import CustomObjectsApi  # type: ignore
from kubernetes.client.exceptions import ApiException  # type: ignore

from fluidos_model_orchestrator.common import Intent
from fluidos_model_orchestrator.common import requires_monitoring
from fluidos_model_orchestrator.common import ResourceFinder
from fluidos_model_orchestrator.common.flavor import build_flavor
from fluidos_model_orchestrator.common.intent import KnownIntent
from fluidos_model_orchestrator.common.model import ModelPredictRequest
from fluidos_model_orchestrator.common.model import ModelPredictResponse
from fluidos_model_orchestrator.common.model import OrchestratorInterface
from fluidos_model_orchestrator.common.prometheus import has_intent_validation_failed
from fluidos_model_orchestrator.common.prometheus import retrieve_metric
from fluidos_model_orchestrator.common.resource import ResourceProvider
from fluidos_model_orchestrator.configuration import CONFIGURATION
from fluidos_model_orchestrator.deployment import redeploy
from fluidos_model_orchestrator.metaorchestrator import validate_with_intents
from fluidos_model_orchestrator.model import _extract_intents
from fluidos_model_orchestrator.model import convert_to_model_request
from fluidos_model_orchestrator.model import get_model_object
from fluidos_model_orchestrator.resources import get_resource_finder
from fluidos_model_orchestrator.rob import DummyResourceProvider


def _is_on_robot(spec: dict[str, Any]) -> bool:
    intents: list[Intent] | None = None

    if spec["kind"] == "Pod":
        intents = _extract_intents(spec["metadata"].get("annotations", {}))

    if spec["kind"] == "Deployment" or spec["kind"] == "Job":
        intents = _extract_intents(spec["metadata"].get("annotations", {}))

    if intents is None:
        return False

    return any(
        intent.name is KnownIntent.robot_preferred for intent in intents
    )


def _get_robots_status() -> dict[str, str | None]:
    metrics = retrieve_metric("robot_status", CONFIGURATION.local_prometheus)
    STATUS: dict[int, str] = {
        0: "idle",
        1: "moving",
        2: "ready",
        3: "charging",
    }

    # with this mapping:
    # scrape_configs:
    # - job_name: 'robot-1-AA'
    #     static_configs:
    #     - targets: ['192.168.75.10:8000']
    # - job_name: 'robot-2-AC'
    #     static_configs:
    #     - targets: ['192.168.75.20:8000']
    robots_status: dict[str, str | None] = {
        "robot-1-AA": None,
        "robot-2-AC": None,
    }

    if metrics:
        for metric in metrics:
            # assume the following structure:
            # robot_status{instance="192.168.75.10:8000", job="robot-1-AA"}
            # robot_status{instance="192.168.75.10:8000", job="robot-2-AC"}
            if metric:
                for robot_name in robots_status.keys():
                    if metric["metric"]["job"] == robot_name:
                        robots_status[robot_name] = STATUS[int(metric["value"][1])]

    return robots_status


def _get_robot_id() -> tuple[str, str]:
    node_id = CONFIGURATION.identity["nodeID"]

    api_client: CustomObjectsApi = CustomObjectsApi(api_client=CONFIGURATION.k8s_client)

    robot_mapping: dict[str, str] = {
        "192.168.75.10:30000": "robot-1-AA",
        "192.168.75.20:30000": "robot-2-AC",
    }

    try:
        flavor_list = api_client.list_namespaced_custom_object(
            group="nodecore.fluidos.eu",
            version="v1alpha1",
            plural="flavors",
            namespace="fluidos")
    except ApiException as e:
        raise e

    for flavor_desc in flavor_list.get("items", []):
        flavor = build_flavor(flavor_desc)

        if flavor.metadata.owner_references["nodeID"] == node_id:
            ip = flavor.metadata.owner_references["ip"]

            return (
                robot_mapping[ip],
                ip
            )
    else:
        raise ValueError()


@kopf.daemon("fluidosdeployments", cancellation_timeout=5)  # type: ignore
async def daemons_for_fluidos_deployment(
        stopped: kopf.DaemonStopped,
        retry: int,
        started: datetime.datetime,
        runtime: datetime.timedelta,
        annotations: bodies.Annotations,
        labels: bodies.Labels,
        body: bodies.Body,
        meta: bodies.Meta,
        spec: dict[str, Any],  # bodies.Spec
        status: bodies.Status,
        uid: str | None,
        name: str,
        namespace: str,
        patch: patches.Patch,
        logger: Logger,
        memo: Any,
        param: Any,
        **kwargs: dict[str, Any]) -> None:
    if not CONFIGURATION.monitor_enabled:
        return

    # check if the spec require monitoring (based on the intents)
    intents_to_monitor: list[Intent] = requires_monitoring(spec, namespace)

    if len(intents_to_monitor) == 0:
        logger.info(f"{namespace}/{name} requires no monitoring")
        return
    else:
        logger.info(f"{namespace}/{name} requires monitoring for {len(intents_to_monitor)} intents")
        for intent in intents_to_monitor:
            logger.debug(f"{namespace}/{name} requires monitoring for {str(intent)} intents")

    # cache objects
    provider_domain: str | None = None
    predictor: OrchestratorInterface | None = None
    finder: ResourceFinder | None = None
    reorchestration_time: datetime.datetime | None = None

    (robot_id, IP) = _get_robot_id()

    CONFIGURATION.host_mapping[IP]

    current_target = None
    best_match: ResourceProvider

    while not stopped.is_set():
        logger.info("Sleeping for %s seconds...", CONFIGURATION.MONITOR_SLEEP_TIME)

        await asyncio.sleep(CONFIGURATION.MONITOR_SLEEP_TIME)

        logger.info("Repeating observation for %s", uid)

        if _is_on_robot(spec):
            # current status chan
            # - navigation pod running on both robots
            # - on "charging" navigation is turned off
            # - when robot is charging it can accept workload from other robot
            # - when robot is idle it can accept workload
            # - when robot is moving it cannot accept workload
            robots_status: dict[str, str | None] = _get_robots_status()

            #  0: "idle",
            #  1: "moving",
            #  2: "ready",
            #  3: "charging",

            # if should offload
            if robots_status[robot_id] in ("moving",):
                # try offloading if other robot can or send to edge
                for id, status in robots_status.items():
                    if id != robot_id and status in ("charging", "idle"):
                        # offload to id
                        offload_target = id
                else:
                    # No one was recepting, send to edge
                    offload_target = "egde"
            else:
                offload_target = id

            # HOST_MAPPING: "192.168.75.10:;192.168.75.20:shy-haze;192.168.75.30:damp-moon"
            label_mapping = {
                "robot-1-AA": "lively-shadow",
                "robot-2-AC": "shy-haze",
                "edge": "damp-moon",
            }

            if offload_target != current_target:
                if label_mapping[offload_target] == label_mapping[robot_id]:
                    s_label = CONFIGURATION.local_node_key
                    s_key = "true"
                else:
                    s_label = CONFIGURATION.remote_node_key
                    s_key = label_mapping[offload_target]

                best_match = DummyResourceProvider(s_label, s_key)
                if not await redeploy(name, namespace, str(spec["kind"]), best_match):
                    print("Something went wrong")
                current_target = offload_target

            continue

        # check if status is set to "completed"
        metaorchestration_status_data: dict[str, Any] = status.get("metaorchestration", {})

        metaorchestration_status = metaorchestration_status_data.get("status", "")

        if metaorchestration_status == "Failure":
            logger.info("%s/%s failed in allocating the system, kill the monitoring process too", namespace, name)
            return

        if metaorchestration_status == "Success":
            logger.info("%s/%s is correctly deployed, check if still valid", namespace, name)

            provider_domain = metaorchestration_status_data["deployed"]["resource_provider"]["flavor"]["spec"]["owner"]["domain"]

            for intent in intents_to_monitor:
                if has_intent_validation_failed(intent, CONFIGURATION.local_prometheus, provider_domain, namespace, name, reorchestration_time):
                    logger.info("%s/%s failed when validating %s", namespace, name, intent.name)
                    break
            else:
                logger.info("%s/%s all intents are still valid", namespace, name)
                continue
        else:
            # anything other than "Failure" or "Success"
            logger.info("%s/%s still being deployed", namespace, name)
            continue

        logger.info("Proceding to reallocate the workload")
        request: ModelPredictRequest | None = convert_to_model_request(spec, namespace)

        if request is None:
            logger.error("Request is not valid, discarding")
            return

        if predictor is None:
            logger.info("Initializing predictor")
            predictor = get_model_object(request)

        prediction: ModelPredictResponse | None = predictor.predict(request, CONFIGURATION.architecture)  # this should use a system defined default, thus from the configuration

        if prediction is None:
            logger.error("Model unable to provide valid prediction")
            return
        else:
            logger.debug(f"Predicted resources for {spec['metadata']['name']}: {prediction}")

        prediction.id = f"{prediction.id}-{uuid4().hex}"

        if finder is None:
            logger.info("Initializing resource finder")
            finder = get_resource_finder(request, prediction)

        resources: list[ResourceProvider] = finder.find_best_match(prediction.to_resource(), namespace)

        logger.debug(f"{resources=}")

        # remove current provider
        resources = [
            resource for resource in resources
            if resource.flavor.spec.owner["domain"] != provider_domain
        ]

        best_matches: list[ResourceProvider] = validate_with_intents(
            predictor.rank_resources(
                resources,
                prediction,
                request
            ), request.intents, logger)

        if not len(best_matches):
            logger.info("Unable to find resource matching requirement")

            return
        else:
            logger.info(f"Retrieved {len(best_matches)} valid resource providers")

        best_match = best_matches[0]

        next_provider_domain = best_match.flavor.spec.owner["domain"]

        logger.info(f"Selected {best_match.id} of type {type(best_match)}")

        if not best_match.acquire(namespace):
            logger.info(f"Unable to acquire {best_match}")

            return

        # find other resources types based on the intents
        if not await redeploy(name, namespace, str(spec["kind"]), best_match):
            logger.info("Unable to deploy")

            return

        provider_domain = next_provider_domain
        reorchestration_time = datetime.datetime.now()

        logger.info("%s/%s Done, sleeping", namespace, name)
