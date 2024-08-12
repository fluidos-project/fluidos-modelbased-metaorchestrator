from datetime import datetime
from datetime import timedelta
try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc
from logging import Logger
from typing import Any

import kopf  # type: ignore


@kopf.on.probe(id="now")  # type: ignore
def healtz_get_current_timestamp(
    *,
    settings: kopf.OperatorSettings,
    retry: int,
    started: datetime,
    runtime: timedelta,
    logger: Logger,
    memo: Any,
    param: Any,
    **kwargs: Any,
) -> str:
    return datetime.now(UTC).isoformat()
