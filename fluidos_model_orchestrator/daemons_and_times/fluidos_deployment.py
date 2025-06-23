import datetime
from logging import Logger
from typing import Any

import kopf  # type: ignore
from kopf._cogs.structs import bodies  # type: ignore
from kopf._cogs.structs import patches  # type: ignore

from fluidos_model_orchestrator.common import Intent
from fluidos_model_orchestrator.common import requires_monitoring
from fluidos_model_orchestrator.common.intent import has_intent_validation_failed
from fluidos_model_orchestrator.configuration import CONFIGURATION


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
                    # find new resource

                    # move workload
                    # ??? deallocate ???

                    stopped.wait(CONFIGURATION.MONITOR_SLEEP_TIME * 2)
                    break
            else:
                logger.info(f"{namespace}/{name} is still valid")
