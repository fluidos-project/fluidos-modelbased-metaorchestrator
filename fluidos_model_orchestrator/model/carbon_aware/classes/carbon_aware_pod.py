from datetime import timedelta, datetime


class CarbonAwarePod:
    def __init__(self, id, deadline_hours, duration, powerConsumption, cpuRequest, ramRequest, storageRequest):
        self.id = id
        self.deadline = processDeadline(self, deadline_hours)
        self.duration = duration
        self.powerConsumption = powerConsumption
        self.cpuRequest = cpuRequest
        self.ramRequest = ramRequest
        self.storageRequest = storageRequest


def processDeadline(self, deadline_hours):
    now = datetime.now()
    delta = timedelta(hours=deadline_hours)
    return now + delta
