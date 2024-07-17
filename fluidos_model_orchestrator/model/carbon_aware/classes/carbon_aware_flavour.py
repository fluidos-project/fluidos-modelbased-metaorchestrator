class CarbonAwareFlavour:
    def __init__(self, id, embodiedCarbon, lifetime, totalCpu, totalRam, totalStorage, forecast):
        self.id = id
        self.embodiedCarbon = embodiedCarbon
        self.lifetime = lifetime
        self.totalCpu = totalCpu
        self.totalRam = totalRam
        self.totalStorage = totalStorage
        self.forecast = forecast