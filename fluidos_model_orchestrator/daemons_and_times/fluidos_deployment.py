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
    if not requires_monitoring(spec, namespace):
        return

    while not stopped:

        stopped.wait(CONFIGURATION.DAEMON_SLEEP_TIME)
