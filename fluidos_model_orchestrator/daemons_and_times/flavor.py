import datetime
from logging import Logger
from typing import Any
from typing import cast

import kopf  # type: ignore
from kopf._cogs.structs import bodies  # type: ignore
from kopf._cogs.structs import patches  # type: ignore

from fluidos_model_orchestrator.common import build_flavor
from fluidos_model_orchestrator.common import ResourceFinder
from fluidos_model_orchestrator.configuration import CONFIGURATION
from fluidos_model_orchestrator.flavor import FlavorK8SliceData
from fluidos_model_orchestrator.model.carbon_aware.forecast_updater import update_local_flavor_forecasted_data
from fluidos_model_orchestrator.resources import get_resource_finder


@kopf.daemon("flavors", cancellation_timeout=1.0)  # type: ignore
async def daemons_for_flavours_observation(
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
    logger.info(f"Running timeseries generation for local flavors only (aka owned by {CONFIGURATION.identity})")

    if not CONFIGURATION.check_identity(spec["owner"]):
        logger.info("Flavor {}{} is not managed locally. Exit", namespace, name)
        return

    finder: ResourceFinder | None = None

    while not stopped:
        stopped.wait(CONFIGURATION.FLAVOR_UPDATE_SLEEP_TIME)
        logger.info(f"Repeating observation for {uid}")
        logger.info(f"Spec: {spec}")
        if stopped:
            logger.info("Stopped by external")
            return
        flavor = build_flavor({
            "metadata": meta,
            "spec": spec
        })

        if namespace is None:
            namespace = "default"

        update_flavor = update_local_flavor_forecasted_data(flavor, namespace)
        if update_flavor is None:
            logger.info("Flavor not updated")
            continue

        if finder is None:
            finder = get_resource_finder()

        finder.update_local_flavor(flavor, cast(FlavorK8SliceData, update_flavor.spec.flavor_type.type_data).properties, namespace)
