from typing import Any


class CarbonAwareFlavour:
    def __init__(self, id: str, embodiedCarbon: Any, lifetime: Any, totalCpu: Any, totalRam: Any, totalStorage: Any, forecast: Any) -> None:
        self.id = id
        self.embodiedCarbon = embodiedCarbon
        self.lifetime = lifetime
        self.totalCpu = totalCpu
        self.totalRam = totalRam
        self.totalStorage = totalStorage
        self.forecast = forecast
