import datetime
import logging
from asyncio import Lock
from typing import Any

import kopf  # type: ignore
from kubernetes import client as k8s_client  # type: ignore
from kubernetes import config as k8s_config  # type: ignore

from .configuration import CONFIGURATION
from .configuration import enrich_configuration


LOCK: Lock


@kopf.on.startup()  # type: ignore
async def configure(settings: kopf.OperatorSettings, retry: int, started: datetime.datetime, runtime: datetime.timedelta, logger: logging.Logger, memo: Any, param: Any, **kwargs: Any) -> None:
    global LOCK

    logger.info("Running initialization functions")

    settings.posting.level = logging.INFO

    LOCK = Lock()

    my_config = k8s_client.Configuration()  # type: ignore
    k8s_config.load_config(client_configuration=my_config)  # type: ignore

    enrich_configuration(CONFIGURATION, settings, param, memo, kwargs, logger, my_config)


@kopf.on.cleanup()  # type: ignore
def cleanup_function(logger: logging.Logger, **kwargs: str) -> None:
    logger.info("Running clean up functionlity")

    print(kwargs)

    pass
