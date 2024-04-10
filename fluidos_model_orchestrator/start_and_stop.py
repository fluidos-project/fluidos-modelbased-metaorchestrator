import datetime
from typing import Any
import kopf
from asyncio import Lock
import logging

from .common import CONFIGURATION
from .common import Configuration

from kubernetes import client
from kubernetes import config as k8config


LOCK: Lock


@kopf.on.startup()  # type: ignore
async def configure(settings: kopf.OperatorSettings, retry: int, started: datetime.datetime, runtime: datetime.timedelta, logger: logging.Logger, memo: Any, param: Any, **kwargs: Any) -> None:
    global LOCK

    logger.info("Running initialization functions")

    settings.posting.level = logging.INFO

    LOCK = Lock()

    enrich_configuration(CONFIGURATION, settings, param, memo, kwargs)


def enrich_configuration(config: Configuration, settings: kopf.OperatorSettings, param: Any, memo: Any, kwargs: dict[str, Any]) -> None:
    logging.info("Enrich default configuration with user provided information")

    my_config = client.Configuration()
    k8config.load_config(client_configuration=my_config)

    config.k8s_client = client.ApiClient(my_config)

    config.node_id = config.k8s_client


@kopf.on.cleanup()  # type: ignore
def cleanup_function(logger: logging.Logger, **kwargs: str) -> None:
    logger.info("Running clean up functionlity")

    print(kwargs)

    pass
