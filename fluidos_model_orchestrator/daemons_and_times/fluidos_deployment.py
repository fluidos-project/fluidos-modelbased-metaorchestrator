import datetime
from logging import Logger
from typing import Any

import kopf  # type: ignore
from kopf._cogs.structs import bodies  # type: ignore
from kopf._cogs.structs import patches  # type: ignore

from fluidos_model_orchestrator.common import Intent
from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.configuration import CONFIGURATION
from fluidos_model_orchestrator.model import convert_to_model_request


def requires_validation(intent: Intent) -> bool:
    return intent.needs_monitoring()


def has_intent_validation_failed(intent: Intent, prometheus_ref: str) -> bool:
    return False


def requires_monitoring(spec: dict[str, Any], namespace: str | None) -> list[Intent]:
    if namespace is None:
        namespace = "default"
    request: ModelPredictRequest | None = convert_to_model_request(spec, namespace)

    if request is None:
        return []

    return [
        intent for intent in request.intents
        if requires_validation(intent)
    ]


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
        name: str | None,
        namespace: str | None,
        patch: patches.Patch,
        logger: Logger,
        memo: Any,
        param: Any,
        **kwargs: dict[str, Any]) -> None:
    # check if the spec require monitoring (based on the intents)
    intents_to_monitor: list[Intent] = requires_monitoring(spec, namespace)

    if len(intents_to_monitor) == 0:
        logger.info(f"{namespace}/{name} requires no monitoring")
        return
    else:
        logger.info(f"{namespace}/{name} requires monitoring for {len(intents_to_monitor)} intents")
        for intent in intents_to_monitor:
            logger.debug(f"{namespace}/{name} requires monitoring for {str(intent)} intents")

    while not stopped:
        stopped.wait(CONFIGURATION.MONITOR_SLEEP_TIME)

        # check if status is set to "completed"
        metaorchestration_status: str = status.get("metaorchestration", {}).get("status", "")

        if metaorchestration_status == "":
            continue  # go to sleep

        if metaorchestration_status == "Failure":
            logger.info(f"{namespace}/{name} failed in allocating the system, kill the monitoring process too")
            return

        if metaorchestration_status == "Success":
            logger.info
            for intent in intents_to_monitor:
                if has_intent_validation_failed(intent, CONFIGURATION.local_prometheus):
                    logger.info(f"{namespace}/{name} failed when validating {intent.name}")

                    # reorchestrate

                    # find the resource
                    stopped.wait(CONFIGURATION.MONITOR_SLEEP_TIME * 2)
                    break
            else:
                logger.info(f"{namespace}/{name} is still valid")
