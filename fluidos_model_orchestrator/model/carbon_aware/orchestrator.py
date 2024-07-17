import logging
from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np
import random

from fluidos_model_orchestrator.common import ModelInterface
from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.common import cpu_to_int
from fluidos_model_orchestrator.common import memory_to_int
from fluidos_model_orchestrator.model.carbon_aware.classes.carbon_aware_flavour import CarbonAwareFlavour
from fluidos_model_orchestrator.model.carbon_aware.classes.carbon_aware_pod import CarbonAwarePod
from fluidos_model_orchestrator.model.carbon_aware.classes.carbon_aware_timeslot import CarbonAwareTimeslot
from fluidos_model_orchestrator.model.carbon_aware.forecast_updater import update_local_flavours_forecasted_data
from fluidos_model_orchestrator.resources import get_resource_finder
from fluidos_model_orchestrator.model.carbon_aware.fakers.weather_forecast_generator import generate_electricity_maps_forecast
from fluidos_model_orchestrator.model.carbon_aware.fakers.workload_prediction_generator import generate_resource_prediction

debug = True

def _debug(message):
    if debug:
        logging.debug(message)

def _is_timeslot_valid(timeslot: CarbonAwareTimeslot, pod: CarbonAwarePod) -> bool:
    return (pod.deadline > timeslot.getStart()) & (datetime.now() <= timeslot.getEnd())


def _check_node_resource(flavour: CarbonAwareFlavour, timeslot: CarbonAwareTimeslot, pod: CarbonAwarePod):
    # todo getResourceUtilizationPrediction(flavour, timeslot) --> Call prediction model to
    # check if there is enough resource left on flavour x at timeslot y for the pod z

    # Temporary implementation:
    cpu_used_prediction = generate_resource_prediction(flavour.totalCpu)
    ram_used_prediction = generate_resource_prediction(flavour.totalRam)

    _debug(f"CPU used prediction: {cpu_used_prediction}")
    _debug(f"RAM used prediction: {ram_used_prediction}")

    _debug(f"CPU left: {flavour.totalCpu - cpu_used_prediction}")
    _debug(f"RAM left: {flavour.totalRam - ram_used_prediction}")
    _debug(f"CPU request: {pod.cpuRequest}")
    _debug(f"RAM request: {pod.ramRequest}")

    if (flavour.totalCpu - cpu_used_prediction) < pod.cpuRequest or (flavour.totalRam - ram_used_prediction) < pod.ramRequest:
        _debug("Node does not have enough resources to accommodate the pod.")
        return False
    _debug("Node has enough resources to accommodate the pod.")
    return True


class CarbonAwareOrchestrator(ModelInterface):
    def predict(self, req: ModelPredictRequest, architecture: str = "arm64") -> ModelPredictResponse:

        deadline = np.nan
        cpuRequest = np.nan
        ramRequest = np.nan
        for intent in req.intents:
            match intent.name.name:
                case "deadline":
                    deadline = int(intent.value)
                    _debug(f"Found deadline from intent file: {deadline}")
                case "cpu":
                    cpuRequest = cpu_to_int(intent.value)
                    _debug(f"Found cpu request from intent file: {cpuRequest}")
                case "memory":
                    ramRequest = memory_to_int(intent.value)
                    _debug(f"Found memory request from intent file: {ramRequest}")
                case _:
                    _debug(f"Intent {intent.name.name} not recognized in Carbon-Aware orchestrator")
        if deadline == np.nan or deadline <= 0 or deadline > 24:
            deadline = np.nan
            logging.exception("Deadline must be provided between ]0;24]")
        if cpuRequest == np.nan or cpuRequest <= 0:
            cpuRequest = np.nan
            logging.exception("CPU request must be provided greater than 0")
        if ramRequest == np.nan or ramRequest <= 0:
            ramRequest = np.nan
            logging.exception("RAM request must be provided greater than 0")

        timeslots = []
        now = datetime.now()
        start_time = now.replace(minute=0, second=0, microsecond=0)
        for i in range(deadline):
            slot_time = start_time + timedelta(hours=i)
            timeslot = CarbonAwareTimeslot(i, slot_time.year, slot_time.month, slot_time.day, slot_time.hour, 2)
            timeslots.append(timeslot)

        _debug(f"Generated timeslots from deadline: {len(timeslots)}")

        resources = get_resource_finder(req.namespace, None).retrieve_all_flavors(req.namespace)
        _debug(f"resources: {resources}")

        flavours = []
        for resource in resources:
            flavours.append(
                CarbonAwareFlavour(
                    resource.id,
                    random.randint(455000, 2502000), # embodiedEmissions (g)
                    4,
                    cpu_to_int(resource.characteristics.cpu),
                    memory_to_int(resource.characteristics.memory),
                    resource.characteristics.persistent_storage,
                    generate_electricity_maps_forecast(deadline)
                ))


        _debug(f"flavours: {flavours}")

        for flavour in flavours:
            _debug(f"flavour x: {flavour.id} {flavour.embodiedCarbon} {flavour.lifetime} {flavour.totalCpu} {flavour.totalRam} {flavour.totalStorage}")

        podToSchedule = CarbonAwarePod(req.id, deadline, 2, 0.03, cpuRequest, ramRequest, 0)

        # --------------------------------- CORE ---------------------------------

        minimal_emissions = np.inf
        best_node = None
        best_timeslot = None

        for ts in timeslots:
            _debug("-------------------- New timeslot iteration --------------------")
            if _is_timeslot_valid(ts, podToSchedule):
                _debug(f"Timeslot {ts.id} is valid.")
                for flavour in flavours:
                    _debug("-------------------- New node iteration --------------------")
                    _debug(f"Checking node {flavour.id}")
                    _debug(f"forecast for this timeslot and this node: {flavour.forecast[ts.id]}")
                    if _check_node_resource(flavour, ts, podToSchedule):
                        forecast = flavour
                        operationalEmissions = (flavour.forecast[ts.id]) * podToSchedule.duration * podToSchedule.powerConsumption # grams, hours, kW
                        embodiedEmissions = ((flavour.embodiedCarbon / (365 * flavour.lifetime * 24)) / (1 + 1)) * podToSchedule.duration

                        totalEmissions = operationalEmissions + embodiedEmissions

                        _debug(f"operationalEmissions: {operationalEmissions} gCO2")
                        _debug(f"embodiedEmissions: {embodiedEmissions} gCO2")
                        _debug(f"Total emissions: {totalEmissions} gCO2")

                        if totalEmissions < minimal_emissions:
                            _debug(f"Updating best node and timeslot.")
                            if best_node is not None and best_timeslot is not None:
                                _debug(f"Previous minimal emissions: {minimal_emissions} gCo2")
                                _debug(f"Previous best timeslot: {best_timeslot.id}")
                                _debug(f"Previous best node: {best_node.id}")

                            _debug(f"New minimal emissions: {totalEmissions} gCo2")
                            _debug(f"New best timeslot: {ts.id}")
                            _debug(f"New best node: {flavour.id}")
                            minimal_emissions = totalEmissions
                            best_node = flavour
                            best_timeslot = ts
                        else:
                            _debug(f"Total emissions for this iteration higher than current minimal emissions.")
                            _debug(f"Total emissions for this iteration: {totalEmissions} gCO2")
                            _debug(f"Current minimum found : {minimal_emissions} gCO2")


        # total_emissions_per_node = np.zeros((2, len(timeslots)))  # 2 nodes, 24 timeslots

        if best_timeslot is None:
            logging.exception("No available timeslot found.")
        if best_node is None:
            logging.exception("No available node found.")

        _debug(f"Best node: {best_node.id}")
        _debug(f"Best timeslot: {best_timeslot.id}")


        update_local_flavours_forecasted_data(req.namespace)

        response = ModelPredictResponse(
            req.id,
            Resource(id=f"carbonaware-{str(uuid4())}", cpu="2n", memory="20Mi", architecture="arm64"))

        return response
