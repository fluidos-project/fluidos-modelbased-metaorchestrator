import kopf  # type: ignore

from fluidos_model_orchestrator.rescheduler.idl import idl_pb2 as hccm
# from fluidos_model_orchestrator.configuration import CONFIGURATION


def _gather_placement(microservices: list[hccm.Microservice]) -> list[hccm.Placement]:
    return []


def _gather_microservices() -> list[hccm.Microservice]:
    return []


def _gather_data_flows(microservices: list[hccm.Microservice]) -> list[hccm.DataFlow] | None:
    return None


def _gather_workload() -> hccm.Workload:
    microservices = _gather_microservices()
    data_flows = _gather_data_flows(microservices)
    placements = _gather_placement(microservices)
    return hccm.Workload(
        microservices, data_flows, placements
    )


def _interrogate_hccm_service(workload: hccm.Workload) -> None:
    pass


@kopf.on.timer("fluidosdeployments", interval=5.0, idle=1, sharp=False)
def rescheduler(spec, **kwargs) -> None:
    optimal_placement = _interrogate_hccm_service(_gather_workload())

    assert optimal_placement is not None
