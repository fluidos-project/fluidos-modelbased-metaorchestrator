import logging
from datetime import datetime
from datetime import timedelta

import numpy as np  # type: ignore

from fluidos_model_orchestrator.common import cpu_to_int
from fluidos_model_orchestrator.common import memory_to_int
from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import OrchestratorInterface
from fluidos_model_orchestrator.common import ResourceProvider
from fluidos_model_orchestrator.model.carbon_aware.classes.carbon_aware_flavour import CarbonAwareFlavour
from fluidos_model_orchestrator.model.carbon_aware.classes.carbon_aware_pod import CarbonAwarePod
from fluidos_model_orchestrator.model.carbon_aware.classes.carbon_aware_timeslot import CarbonAwareTimeslot
from fluidos_model_orchestrator.model.carbon_aware.fakers.workload_prediction_generator import generate_resource_prediction

debug = True


def _debug(message: str) -> None:
    if debug:
        print(message)


def _is_timeslot_valid(timeslot: CarbonAwareTimeslot, pod: CarbonAwarePod) -> bool:
    return (pod.deadline > timeslot.getStart()) & (datetime.now() <= timeslot.getEnd())


def _check_node_resource(flavour: CarbonAwareFlavour, timeslot: CarbonAwareTimeslot, pod: CarbonAwarePod) -> bool:
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

    if (flavour.totalCpu - cpu_used_prediction) < pod.cpuRequest or (
            flavour.totalRam - ram_used_prediction) < pod.ramRequest:
        _debug("Node does not have enough resources to accommodate the pod.")
        return False
    _debug("Node has enough resources to accommodate the pod.")
    return True


class CarbonAwareOrchestrator(OrchestratorInterface):
    def rank_resource(self, providers: list[ResourceProvider], prediction: ModelPredictResponse,
                      request: ModelPredictRequest) -> list[ResourceProvider]:

        _debug(f"ModelPredictRequest pod_request: {request.pod_request}")

        deadline = np.nan
        cpuRequest = np.nan
        ramRequest = np.nan
        for intent in request.intents:
            match intent.name.name:
                case "deadline":
                    deadline = int(intent.value)
                    deadline += 1
                    _debug(f"Found deadline from intent file (+1): {deadline}")
                case "cpu":
                    cpuRequest = cpu_to_int(intent.value)
                    _debug(f"Found cpu request from intent file: {cpuRequest}")
                case "memory":
                    ramRequest = memory_to_int(intent.value)
                    _debug(f"Found memory request from intent file: {ramRequest}")
                case _:
                    _debug(f"Intent {intent.name.name} not recognized in Carbon-Aware orchestrator")
        if deadline == np.nan or deadline <= 0 or deadline > 24:
            logging.exception("Deadline must be provided between ]0;24]")
            return []
        if cpuRequest == np.nan or cpuRequest <= 0:
            logging.exception("CPU request must be provided greater than 0")
            return []
        if ramRequest == np.nan or ramRequest <= 0:
            logging.exception("RAM request must be provided greater than 0")
            return []

        timeslots: list[CarbonAwareTimeslot] = []
        now = datetime.now()
        start_time = now.replace(minute=0, second=0, microsecond=0)
        for i in range(int(deadline)):
            slot_time = start_time + timedelta(hours=i)
            timeslot = CarbonAwareTimeslot(i, slot_time.year, slot_time.month, slot_time.day, slot_time.hour, 2)
            timeslots.append(timeslot)

        _debug(f"Generated timeslots from deadline: {len(timeslots)}")

        flavours = []
        for provider in providers:
            flavor = provider.flavor
            _debug(f"provider ID: {provider.id}")
            _debug(f"flavor ID: {provider.flavor.id}")
            _debug(f"flavor optional_fields: {flavor.optional_fields}")
            flavours.append(
                CarbonAwareFlavour(
                    flavor.id,
                    flavor.optional_fields.get("embodied"),
                    4,
                    cpu_to_int(flavor.characteristics.cpu),
                    memory_to_int(flavor.characteristics.memory),
                    flavor.characteristics.persistent_storage,
                    flavor.optional_fields.get("operational")
                ))

        logging.debug(f"flavours: {flavours}")

        for flavour in flavours:
            _debug(
                f"flavour x: {flavour.id} {flavour.embodiedCarbon} {flavour.lifetime} {flavour.totalCpu} {flavour.totalRam} {flavour.totalStorage}")

        podToSchedule = CarbonAwarePod(request.id, deadline, 2, np.nan, cpuRequest, ramRequest, 0)

        # --------------------------------- CORE ---------------------------------

        minimal_emissions = np.inf
        best_node = None
        best_timeslot: CarbonAwareTimeslot | None = None

        for ts in timeslots:
            _debug("-------------------- New timeslot iteration --------------------")
            if _is_timeslot_valid(ts, podToSchedule):
                _debug(f"Timeslot {ts.id} is valid.")
                for flavour in flavours:
                    _debug("-------------------- New node iteration --------------------")
                    _debug(f"Checking node {flavour.id}")
                    _debug(f"forecast for this timeslot and this node: {flavour.forecast[ts.id]}")
                    if _check_node_resource(flavour, ts, podToSchedule):
                        podToSchedule.powerConsumption = (podToSchedule.cpuRequest / flavour.totalCpu) * 0.3
                        _debug(f"cpuRequest: {podToSchedule.cpuRequest}")
                        _debug(f"totalCpu: {flavour.totalCpu}")
                        _debug(f"powerConsumption of pod: {podToSchedule.powerConsumption}")
                        operationalEmissions = (flavour.forecast[ts.id]) * podToSchedule.duration * podToSchedule.powerConsumption  # grams, hours, kW
                        embodiedEmissions = ((flavour.embodiedCarbon / (365 * flavour.lifetime * 24)) / (1 + 1)) * podToSchedule.duration

                        totalEmissions = operationalEmissions + embodiedEmissions

                        _debug(f"operationalEmissions: {operationalEmissions} gCO2")
                        _debug(f"embodiedEmissions: {embodiedEmissions} gCO2")
                        _debug(f"Total emissions: {totalEmissions} gCO2")

                        if totalEmissions < minimal_emissions:
                            _debug("Updating best node and timeslot.")
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
                            _debug("Total emissions for this iteration higher than current minimal emissions.")
                            _debug(f"Total emissions for this iteration: {totalEmissions} gCO2")
                            _debug(f"Current minimum found : {minimal_emissions} gCO2")

        if best_timeslot is None:
            logging.exception("No available timeslot found.")
            raise RuntimeError("Failing")
        if best_node is None:
            logging.exception("No available node found.")
            raise RuntimeError("Failing")

        _debug(f"Best node: {best_node.id}")
        _debug(f"Best timeslot (and prediction delay): {best_timeslot.id}")
        prediction.delay = best_timeslot.id

        bestProvider = []
        for provider in providers:
            if provider.id == best_node.id:
                bestProvider.append(provider)
                return bestProvider  # return list of 1 element with best node
        return []

    def predict(self, req: ModelPredictRequest, architecture: str = "arm64") -> ModelPredictResponse | None:
        return None
