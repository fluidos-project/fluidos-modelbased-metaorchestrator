from fluidos_model_orchestrator.common import Flavor
from fluidos_model_orchestrator.common import FlavorCharacteristics
from fluidos_model_orchestrator.common import FlavorType
from fluidos_model_orchestrator.common import Intent
from fluidos_model_orchestrator.common import KnownIntent
from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.common import ResourceProvider
from fluidos_model_orchestrator.model.carbon_aware.orchestrator import CarbonAwareOrchestrator
from fluidos_model_orchestrator.resources.rear.local_resource_provider import LocalResourceProvider


def test_basic_ranking() -> None:
    providers: list[ResourceProvider] = [
        LocalResourceProvider("123", Flavor("123", FlavorType.K8SLICE, FlavorCharacteristics("1", "amd64", "10", "10Gi"), {}, "provider1", {}, {"operational": [100, 120, 130, 150, 120, 110, 100, 90, 130, 150, 100, 80, 180, 40, 80, 130], "embodied": 1300})),
        LocalResourceProvider("122", Flavor("122", FlavorType.K8SLICE, FlavorCharacteristics("1", "amd64", "10", "10Gi"), {}, "provider2", {}, {"operational": [90, 110, 120, 140, 110, 100, 90, 80, 120, 140, 90, 70, 170, 30, 70, 120], "embodied": 1200})),
    ]
    request = ModelPredictRequest(
        id="req1",
        namespace="default",
        pod_request={},
        container_image_embeddings=[],
        intents=[Intent(name=KnownIntent.deadline, value="1")],
    )
    prediction = ModelPredictResponse(
        "resp-123",
        Resource(id="res-123", architecture="amd64", cpu="1", memory="1"),
        delay=1
    )

    orchestrator = CarbonAwareOrchestrator()

    ranked = orchestrator.rank_resource(providers, prediction, request)

    assert ranked is not None
    assert len(ranked) == 1
