from datetime import timedelta, datetime


class CarbonAwareTimeslot:
    def __init__(self, id, startYear, startMonth, startDay, startHour, length):
        self.id = id
        self.start = datetime(startYear, startMonth, startDay, startHour)
        self.length = timedelta(hours=length)

    def getEnd(self):
        return self.start + self.length

    def getStart(self):
        return self.start