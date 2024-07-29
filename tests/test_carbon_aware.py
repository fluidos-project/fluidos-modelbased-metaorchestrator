from fluidos_model_orchestrator.common import Flavor
from fluidos_model_orchestrator.common import FlavorCharacteristics
from fluidos_model_orchestrator.common import FlavorType
from fluidos_model_orchestrator.common import Intent
from fluidos_model_orchestrator.common import KnownIntent
from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.model.carbon_aware.orchestrator import CarbonAwareOrchestrator
from fluidos_model_orchestrator.resources.rear.local_resource_provider import LocalResourceProvider


def test_basic_ranking() -> None:
    orchestrator = CarbonAwareOrchestrator()

    ranked = orchestrator.rank_resource([
        LocalResourceProvider("123", Flavor("123", FlavorType.K8SLICE, FlavorCharacteristics("1", "arm64", "10", "10"), {}, "provider1")),
        LocalResourceProvider("122", Flavor("122", FlavorType.K8SLICE, FlavorCharacteristics("1", "arm64", "10", "10"), {}, "provider2")),
    ], ModelPredictResponse(
        "resp-123",
        Resource(id="res-123", architecture="arm64", cpu="1", memory="1"),
        delay=1
    ),
        ModelPredictRequest(
            id="req1",
            namespace="default",
            pod_request={},
            container_image_embeddings=[],
            intents=[Intent(name=KnownIntent.deadline, value="2")],
    ))

    assert ranked is not None
    assert len(ranked) == 1
