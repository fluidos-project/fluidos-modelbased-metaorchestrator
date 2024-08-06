from __future__ import annotations

from datetime import datetime
from datetime import timedelta


class CarbonAwarePod:
    def __init__(self, id: str, deadline_hours: float, duration: int, powerConsumption: float, cpuRequest: float, ramRequest: float, storageRequest: int):
        self.id = id
        self.deadline = processDeadline(deadline_hours)
        self.duration = duration
        self.powerConsumption = powerConsumption
        self.cpuRequest = cpuRequest
        self.ramRequest = ramRequest
        self.storageRequest = storageRequest


def processDeadline(deadline_hours: float) -> datetime:
    now = datetime.now()
    delta = timedelta(hours=deadline_hours)
    return now + delta
