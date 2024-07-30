import asyncio
from logging import Logger
from typing import Any
from fluidos_model_orchestrator.common import build_flavor
from fluidos_model_orchestrator.model.carbon_aware.forecast_updater import update_local_flavor_forecasted_data

import kopf  # type: ignore

from fluidos_model_orchestrator.configuration import CONFIGURATION


@kopf.daemon("flavours", cancellation_timeout=1.0)  # type: ignore
async def daemons_for_flavours_observation(uid: str | None, stopped: bool, logger: Logger, spec: dict[str, Any], namespace: str, **kwargs: dict[str, Any]) -> None:
    try:
        logger.info(f"Running timeseries generation for local flavors only (aka owned by {CONFIGURATION.identity})")
        if not CONFIGURATION.check_identity(spec["owner"]):
            logger.info("Not locally managed flavor, exit")
            return

        while True:
            logger.info(f"Repeating observation for {uid}")
            logger.info(f"Spec: {spec}")

            flavor = build_flavor(spec)
            update_local_flavor_forecasted_data(flavor, namespace)

            if stopped:
                logger.info("Stopped by external")
                return

            await asyncio.sleep(CONFIGURATION.DAEMON_SLEEP_TIME)
    except asyncio.CancelledError:
        logger.info(f"We are done for {uid}. Exiting")
