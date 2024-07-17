from fluidos_model_orchestrator.model.carbon_aware.classes.carbon_aware_flavour import CarbonAwareFlavour
from fluidos_model_orchestrator.model.carbon_aware.classes.carbon_aware_timeslot import CarbonAwareTimeslot
import random

def generate_workload_prediction(timeslots: list[CarbonAwareTimeslot], flavour: CarbonAwareFlavour):
    cpu_left_prediction = []
    ram_left_prediction = []
    storage_left_prediction = []
    energy_consumption_prediction = []
    for ts in timeslots:
        cpu_left_prediction = _generate_resource_prediction(100)
        ram_left_prediction = _generate_resource_prediction(100)
        storage_left_prediction = _generate_resource_prediction(100)
        energy_consumption_prediction = _generate_resource_prediction(100)
    return cpu_left_prediction, ram_left_prediction, storage_left_prediction

def generate_resource_prediction(totalResource):
    return random.randint(0, totalResource) # Cores, RAM, Storage