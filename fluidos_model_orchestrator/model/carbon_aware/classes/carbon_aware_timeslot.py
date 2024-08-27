from datetime import datetime
from datetime import timedelta


class CarbonAwareTimeslot:
    def __init__(self, id: int, startYear: int, startMonth: int, startDay: int, startHour: int, length: int) -> None:
        self.id = id
        self.start = datetime(startYear, startMonth, startDay, startHour)
        self.length = timedelta(hours=length)

    def getEnd(self) -> datetime:
        return self.start + self.length

    def getStart(self) -> datetime:
        return self.start
