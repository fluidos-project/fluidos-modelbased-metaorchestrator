import random
from typing import List

from fluidos_model_orchestrator.model.carbon_aware.classes.carbon_aware_timeslot import CarbonAwareTimeslot


def generate_electricity_maps_forecast(deadline: int):
    avgCarbonIntensities = []
    for hour in range(deadline):
        avgCarbonIntensities.append(random.randint(10, 1000))
    return avgCarbonIntensities
