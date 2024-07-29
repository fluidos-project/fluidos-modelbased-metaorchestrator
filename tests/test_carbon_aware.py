from build.lib.fluidos_model_orchestrator.resources import LocalResourceProvider
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.model.carbon_aware.orchestrator import CarbonAwareOrchestrator


def test_basic_ranking() -> None:
    orchestrator = CarbonAwareOrchestrator()

    ranked = orchestrator.rank_resource([
        LocalResourceProvider(),
        LocalResourceProvider(),
    ], ModelPredictResponse())

    assert ranked is not None
    assert len(ranked) == 1
