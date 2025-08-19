import asyncio
import datetime
from logging import Logger
from typing import Any
from uuid import uuid4

import kopf  # type: ignore
from kopf._cogs.structs import bodies  # type: ignore
from kopf._cogs.structs import patches  # type: ignore

from fluidos_model_orchestrator.common import Intent
from fluidos_model_orchestrator.common import requires_monitoring
from fluidos_model_orchestrator.common import ResourceFinder
from fluidos_model_orchestrator.common.intent import KnownIntent
from fluidos_model_orchestrator.common.model import ModelPredictRequest
from fluidos_model_orchestrator.common.model import ModelPredictResponse
from fluidos_model_orchestrator.common.model import OrchestratorInterface
from fluidos_model_orchestrator.common.prometheus import has_intent_validation_failed
from fluidos_model_orchestrator.common.resource import ResourceProvider
from fluidos_model_orchestrator.configuration import CONFIGURATION
from fluidos_model_orchestrator.deployment import redeploy
from fluidos_model_orchestrator.metaorchestrator import validate_with_intents
from fluidos_model_orchestrator.model import _extract_intents
from fluidos_model_orchestrator.model import convert_to_model_request
from fluidos_model_orchestrator.model import get_model_object
from fluidos_model_orchestrator.resources import get_resource_finder


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
            # - when robot is moving it cannot workload
            pass

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
