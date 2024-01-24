import datetime
from typing import Any
import kopf
from asyncio import Lock
import logging

from .common import configuration
from .common import Configuration


LOCK: Lock


@kopf.on.startup()
async def configure(logger: logging.Logger, settings: kopf.OperatorSettings, param, retry: int, started: datetime.datetime, runtime: datetime.timedelta, memo: dict, **kwargs):
    global LOCK

    logger.info("Running initialization functions")

    settings.posting.level = logging.INFO

    LOCK = Lock()

    enrich_configuration(configuration, settings, param, memo, kwargs)


def enrich_configuration(config: Configuration, settings: kopf.OperatorSettings, param: Any, memo: dict, kwargs):
    logging.info("Enrich default configuration with user provided information")

    pass


@kopf.on.cleanup()
def cleanup_function(logger: logging.Logger, **kwargs):
    logger.info("Running clean up functionlity")

    print(kwargs)

    pass
