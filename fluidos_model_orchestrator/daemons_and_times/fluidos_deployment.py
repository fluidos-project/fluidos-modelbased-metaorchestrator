import datetime
from logging import Logger
from typing import Any

import kopf  # type: ignore
from kopf._cogs.structs import bodies  # type: ignore
from kopf._cogs.structs import patches  # type: ignore

from fluidos_model_orchestrator.configuration import CONFIGURATION


def requires_monitoring(spec: dict[str, Any]) -> bool:
    return True


@kopf.daemon("fluidosdeployments")  # type: ignore
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
    if not requires_monitoring(spec):
        return

    while not stopped:

        stopped.wait(CONFIGURATION.DAEMON_SLEEP_TIME)
