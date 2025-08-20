import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import requests

from fluidos_model_orchestrator.common.intent import Intent
# from datetime import datetime
# from datetime import timedelta


logger = logging.getLogger(__name__)


def retrieve_metric(metric: str, host: str) -> list[dict[str, Any]]:
    try:
        # now = datetime.now().replace(microsecond=0)
        # before = timedelta(minutes=15)

        # start = now - before

        query_params = {
            # "start": f"{start.isoformat()}Z",
            # "end": f"{now.isoformat()}Z",
            # "step": "2m",
            # "limit": 100,
            "query": metric,
        }
        headers = {"Content-Type": "application/json"}

        url = f"{host}/api/v1/query"

        response = requests.get(url, params=query_params, headers=headers)  # type: ignore[arg-type]

        if response.status_code // 100 == 2:
            data = response.json()
            if data["status"] != "success":
                logger.error("Failed for not clear reason")
                return []
            return data.get("data", {}).get("result", [])

        elif response.status_code == 400:
            logger.error("Bad Request when parameters are missing or incorrect.")
        elif response.status_code == 422:
            logger.error("Unprocessable Entity when an expression can't be executed (RFC4918).")
        elif response.status_code == 503:
            logger.error("Service Unavailable.")

    except requests.exceptions.RequestException as e:
        # logger.error("Something went very wrong")
        logger.error("Something went very wrong", exc_info=e)

    return []


def has_intent_validation_failed(intent: Intent, prometheus_ref: str, domain: str, namespace: str, name: str, last_reorchestration: datetime | None) -> bool:
    if not intent.name.needs_monitoring():
        logger.info("Not to be monitored, assuming still valid")
        return False
    metric: Callable[[list[str]], str] | None = intent.name.metric_name()
    if metric is None:
        logger.info("Not to be monitored, assuming still valid")
        return False

    metric_to_query = metric([domain, namespace, name])
    data = retrieve_metric(
        metric_to_query,  # noop to make mypy happy
        prometheus_ref)

    if last_reorchestration is not None:
        data = _remove_old_metrics(data, last_reorchestration)

    return not intent.name.validate_monitoring(intent.value, data)


def _remove_old_metrics(data: list[dict[str, Any]], t: datetime) -> list[dict[str, Any]]:
    timestamp = t.timestamp()

    for i in range(len(data)):
        values = data[i]["values"]

        data[i]["values"] = [
            value for value in values
            if value[i] > timestamp
        ]

    return data
