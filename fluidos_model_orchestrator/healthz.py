import kopf
import datetime


@kopf.on.probe(id="now")  # type: ignore
def healtz_get_current_timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()
