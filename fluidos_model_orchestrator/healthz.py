import datetime

import kopf  # type: ignore


@kopf.on.probe(id="now")
def healtz_get_current_timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()
