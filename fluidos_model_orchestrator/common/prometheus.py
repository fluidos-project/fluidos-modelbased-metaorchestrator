import logging
from datetime import datetime
from datetime import timedelta
from typing import Any

import requests

from fluidos_model_orchestrator.common.intent import Intent


logger = logging.getLogger(__name__)


def retrieve_metric(metric: str, host: str) -> dict[str, Any] | None:
    try:
        now = datetime.now().replace(microsecond=0)
        before = timedelta(minutes=15)

        start = now - before

        query_params = {
            "start": f"{start.isoformat()}Z",
            "end": f"{now.isoformat()}Z",
            "step": "2m",
            "limit": 100,
            "query": metric,
        }
        headers = {"Content-Type": "application/json"}

        response = requests.get(f"{host}/api/v1/query_range", params=query_params, headers=headers)  # type: ignore[arg-type]

        print(response.content)

        if response.status_code // 100 == 2:
            data = response.json()
            if data["status"] != "success":
                logger.error("Failed for not clear reason")
                return None
            return data["data"]["result"]

        elif response.status_code == 400:
            logger.error("Bad Request when parameters are missing or incorrect.")
        elif response.status_code == 422:
            logger.error("Unprocessable Entity when an expression can't be executed (RFC4918).")
        elif response.status_code == 503:
            logger.error("Service Unavailable.")

    except requests.exceptions.RequestException as e:
        # logger.error("Something went very wrong")
        logger.error("Something went very wrong", exc_info=e)

    return None


def has_intent_validation_failed(intent: Intent, prometheus_ref: str) -> bool:
    if intent.name.needs_monitoring():
        data = retrieve_metric(
            str(intent.name._metric_name),  # noop to make mypy happy
            prometheus_ref)

        return intent.name.validate_monitoring(intent.value, data)
    return False
