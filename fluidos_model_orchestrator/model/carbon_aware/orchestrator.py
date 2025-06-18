import logging
from datetime import datetime
from datetime import timedelta
from typing import cast

import numpy as np  # type: ignore

from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import ResourceProvider
from fluidos_model_orchestrator.common.flavor import FlavorK8SliceData
from fluidos_model_orchestrator.common.intent import KnownIntent
from fluidos_model_orchestrator.common.model import OrchestratorInterface
from fluidos_model_orchestrator.common.resource import cpu_to_int
from fluidos_model_orchestrator.common.resource import memory_to_int
from fluidos_model_orchestrator.model.carbon_aware.classes.carbon_aware_flavour import CarbonAwareFlavour
from fluidos_model_orchestrator.model.carbon_aware.classes.carbon_aware_pod import CarbonAwarePod
from fluidos_model_orchestrator.model.carbon_aware.classes.carbon_aware_timeslot import CarbonAwareTimeslot
from fluidos_model_orchestrator.model.carbon_aware.fakers.workload_prediction_generator import generate_resource_prediction

logger = logging.getLogger(__name__)


def _is_timeslot_valid(timeslot: CarbonAwareTimeslot, pod: CarbonAwarePod) -> bool:
    return (pod.deadline > timeslot.getStart()) & (datetime.now() <= timeslot.getEnd())


def _check_node_resource(flavour: CarbonAwareFlavour, timeslot: CarbonAwareTimeslot, pod: CarbonAwarePod) -> bool:
    # todo getResourceUtilizationPrediction(flavour, timeslot) --> Call prediction model to
    # check if there is enough resource left on flavour x at timeslot y for the pod z
    # Temporary implementation:
    cpu_used_prediction = generate_resource_prediction(flavour.totalCpu)
    ram_used_prediction = generate_resource_prediction(flavour.totalRam)

    logger.debug(f"CPU used prediction: {cpu_used_prediction}")
    logger.debug(f"RAM used prediction: {ram_used_prediction}")

    logger.debug(f"CPU left: {flavour.totalCpu - cpu_used_prediction}")
    logger.debug(f"RAM left: {flavour.totalRam - ram_used_prediction}")
    logger.debug(f"CPU request: {pod.cpuRequest}")
    logger.debug(f"RAM request: {pod.ramRequest}")

    if (flavour.totalCpu - cpu_used_prediction) < pod.cpuRequest or (
            flavour.totalRam - ram_used_prediction) < pod.ramRequest:
        logger.debug("Node does not have enough resources to accommodate the pod.")
        return False
    logger.debug("Node has enough resources to accommodate the pod.")
    return True


class CarbonAwareOrchestrator(OrchestratorInterface):
    def rank_resources(self, providers: list[ResourceProvider], prediction: ModelPredictResponse,
                       request: ModelPredictRequest) -> list[ResourceProvider]:
        logger.debug(f"ModelPredictRequest pod_request: {request.pod_request}")

        deadline = np.nan
        cpuRequest = 1
        ramRequest = 1
        for intent in request.intents:
            if intent.name is KnownIntent.max_delay:
                deadline = int(intent.value)
                deadline += 1
                logger.info(f"Found deadline from intent file (+1): {deadline}")
            elif intent.name is KnownIntent.cpu:
                cpuRequest = cpu_to_int(intent.value)
                logger.debug(f"Found cpu request from intent file: {cpuRequest}")
            elif intent.name is KnownIntent.memory:
                ramRequest = memory_to_int(intent.value)
                logger.debug(f"Found memory request from intent file: {ramRequest}")
            else:
                logger.debug(f"Intent {intent.name.name} not recognized in Carbon-Aware orchestrator")
        if deadline == np.nan or deadline <= 0 or deadline > 24:
            logging.exception("Deadline must be provided between ]0;24]")
            return []
        if cpuRequest == np.nan or cpuRequest <= 0:
            logging.exception("CPU request must be provided greater than 0")
            return []
        if ramRequest == np.nan or ramRequest <= 0:
            logging.exception("RAM request must be provided greater than 0")
            return []

        now = datetime.now()
        start_time = now.replace(minute=0, second=0, microsecond=0)

        timeslots: list[CarbonAwareTimeslot] = [
            CarbonAwareTimeslot(i, slot_time.year, slot_time.month, slot_time.day, slot_time.hour, 2)
            for (i, slot_time) in [
                (i, start_time + timedelta(hours=i)) for i in range(int(deadline))
            ]
        ]

        logger.debug(f"Generated timeslots from deadline: {len(timeslots)}")

        flavours: list[CarbonAwareFlavour] = []
        for provider in providers:
            flavor = provider.flavor
            type_data = cast(FlavorK8SliceData, flavor.spec.flavor_type.type_data)
            logger.debug(f"provider ID: {provider.id}")
            logger.debug(f"flavor ID: {provider.flavor.metadata.name}")
            logger.debug(f"flavor optional_fields: {type_data.properties}")
            operational = type_data.properties.get("carbon-footprint", {}).get("operational", None)

            if operational is None:
                logger.info(f"Skipping flavor {provider.flavor.metadata.name} from provider {provider.id} as with no carbon information")
                continue

            flavours.append(
                CarbonAwareFlavour(
                    flavor.metadata.name,
                    type_data.properties.get("carbon-footprint", {}).get("embodied", 0),
                    4,
                    cpu_to_int(type_data.characteristics.cpu),
                    memory_to_int(type_data.characteristics.memory),
                    type_data.characteristics.storage,
                    operational
                )
            )

        logging.debug(f"flavours: {flavours}")

        for flavour in flavours:
            logger.debug(
                f"flavour x: {flavour.id} {flavour.embodiedCarbon} {flavour.lifetime} {flavour.totalCpu} {flavour.totalRam} {flavour.totalStorage}")

        podToSchedule = CarbonAwarePod(request.id, deadline, 2, np.nan, cpuRequest, ramRequest, 0)

        # --------------------------------- CORE ---------------------------------

        minimal_emissions = np.inf
        best_node = None
        best_timeslot: CarbonAwareTimeslot | None = None

        for ts in timeslots:
            logger.debug("-------------------- New timeslot iteration --------------------")
            if _is_timeslot_valid(ts, podToSchedule):
                logger.debug(f"Timeslot {ts.id} is valid.")
                for flavour in flavours:
                    logger.debug("-------------------- New node iteration --------------------")
                    logger.debug(f"Checking node {flavour.id}")
                    logger.debug(f"forecast for this timeslot and this node: {flavour.forecast[ts.id]}")
                    if _check_node_resource(flavour, ts, podToSchedule):
                        podToSchedule.powerConsumption = (podToSchedule.cpuRequest / flavour.totalCpu) * 0.3
                        logger.debug(f"cpuRequest: {podToSchedule.cpuRequest}")
                        logger.debug(f"totalCpu: {flavour.totalCpu}")
                        logger.debug(f"powerConsumption of pod: {podToSchedule.powerConsumption}")
                        operationalEmissions = (flavour.forecast[ts.id]) * podToSchedule.duration * podToSchedule.powerConsumption  # grams, hours, kW
                        embodiedEmissions = ((flavour.embodiedCarbon / (365 * flavour.lifetime * 24)) / (1 + 1)) * podToSchedule.duration

                        totalEmissions = operationalEmissions + embodiedEmissions

                        logger.debug(f"operationalEmissions: {operationalEmissions} gCO2")
                        logger.debug(f"embodiedEmissions: {embodiedEmissions} gCO2")
                        logger.debug(f"Total emissions: {totalEmissions} gCO2")

                        if totalEmissions < minimal_emissions:
                            logger.debug("Updating best node and timeslot.")
                            if best_node is not None and best_timeslot is not None:
                                logger.debug(f"Previous minimal emissions: {minimal_emissions} gCo2")
                                logger.debug(f"Previous best timeslot: {best_timeslot.id}")
                                logger.debug(f"Previous best node: {best_node.id}")

                            logger.debug(f"New minimal emissions: {totalEmissions} gCo2")
                            logger.debug(f"New best timeslot: {ts.id}")
                            logger.debug(f"New best node: {flavour.id}")
                            minimal_emissions = totalEmissions
                            best_node = flavour
                            best_timeslot = ts
                        else:
                            logger.debug("Total emissions for this iteration higher than current minimal emissions.")
                            logger.debug(f"Total emissions for this iteration: {totalEmissions} gCO2")
                            logger.debug(f"Current minimum found : {minimal_emissions} gCO2")

        if best_timeslot is None:
            logging.exception("No available timeslot found.")
            return providers

        if best_node is None:
            logging.exception("No available node found.")
            return providers

        logger.info(f"Best node: {best_node.id}")
        logger.info(f"Best timeslot (and prediction delay): {best_timeslot.id}")
        prediction.delay = best_timeslot.id

        for provider in providers:
            if provider.flavor.metadata.name == best_node.id:
                return [provider]  # return list of 1 element with best node

        return providers

    def predict(self, req: ModelPredictRequest, architecture: str = "amd64") -> ModelPredictResponse | None:
        return None
