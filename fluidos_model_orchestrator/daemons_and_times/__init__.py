from __future__ import annotations

import asyncio
from logging import Logger
from typing import Any

import kopf  # type: ignore

from fluidos_model_orchestrator.configuration import CONFIGURATION


def _check_identity(local_identity: dict[str, str], flavor_owner: dict[str, str]) -> bool:
    return all(
        local_identity[key] == flavor_owner.get(key, "")
        for key in local_identity.keys()
    )


@kopf.daemon("flavours", cancellation_timeout=1.0)  # type: ignore
async def daemons_for_flavours_observation(uid: str | None, stopped: bool, logger: Logger, spec: dict[str, Any], **kwargs: dict[str, Any]) -> None:
    try:
        logger.info(f"Running timeseries generation for local flavors only (aka owned by {CONFIGURATION.identity})")
        if not _check_identity(CONFIGURATION.identity, spec["owner"]):
            logger.info("Not locally managed flavor, exit")
            return

        while True:
            logger.info(f"Repeating observation for {uid}")
            logger.info(f"Spec: {spec}")
            if stopped:
                logger.info("Stopped by external")
                return

            await asyncio.sleep(CONFIGURATION.DAEMON_SLEEP_TIME)
    except asyncio.CancelledError:
        logger.info(f"We are done for {uid}. Exiting")
