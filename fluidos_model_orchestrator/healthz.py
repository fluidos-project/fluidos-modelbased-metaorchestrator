import kopf
from datetime import datetime


@kopf.on.probe(id="now")
def healtz_get_current_timestamp(**_) -> str:
    return datetime.utcnow().isoformat()
