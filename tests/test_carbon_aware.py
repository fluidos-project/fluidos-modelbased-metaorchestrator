from fluidos_model_orchestrator.common import Flavor
from fluidos_model_orchestrator.common import FlavorCharacteristics
from fluidos_model_orchestrator.common import FlavorK8SliceData
from fluidos_model_orchestrator.common import FlavorMetadata
from fluidos_model_orchestrator.common import FlavorSpec
from fluidos_model_orchestrator.common import FlavorType
from fluidos_model_orchestrator.common import FlavorTypeData
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
        LocalResourceProvider(
            "123",
            Flavor(
                metadata=FlavorMetadata(name="123", owner_references={}),
                spec=FlavorSpec(
                    availability=True,
                    flavor_type=FlavorTypeData(
                        type_data=FlavorK8SliceData(
                            characteristics=FlavorCharacteristics(
                                architecture="amd64",
                                cpu="1",
                                memory="10Gi",
                                gpu=1
                            ),
                            policies={},
                            properties={
                                "carbon-footprint": {
                                    "operational": [100, 120, 130, 150, 120, 110, 100, 90, 130, 150, 100, 80, 180, 40, 80, 130],
                                    "embodied": 1300
                                }
                            }
                        ),
                        type_identifier=FlavorType.K8SLICE,
                    ),
                    network_property_type="",
                    owner={},
                    providerID="",
                    price={},
                    location={}
                )
            ),
        ),
        LocalResourceProvider(
            "122",
            Flavor(
                metadata=FlavorMetadata(name="123", owner_references={}),
                spec=FlavorSpec(
                    availability=True,
                    flavor_type=FlavorTypeData(
                        type_data=FlavorK8SliceData(
                            characteristics=FlavorCharacteristics(
                                architecture="amd64",
                                cpu="1",
                                memory="10Gi",
                                gpu=1
                            ),
                            policies={},
                            properties={
                                "carbon-footprint": {
                                    "operational": [90, 110, 120, 140, 110, 100, 90, 80, 120, 140, 90, 70, 170, 30, 70, 120],
                                    "embodied": 1200
                                }
                            }
                        ),
                        type_identifier=FlavorType.K8SLICE,
                    ),
                    network_property_type="",
                    owner={},
                    providerID="",
                    price={},
                    location={}
                )
            ),
        ),
    ]
    request = ModelPredictRequest(
        id="req1",
        namespace="default",
        pod_request={},
        container_image_embeddings=[],
        intents=[
            Intent(name=KnownIntent.max_delay, value="1"),
        ],
    )
    prediction = ModelPredictResponse(
        "resp-123",
        Resource(id="res-123", architecture="amd64", cpu="1", memory="1"),
        delay=1
    )

    orchestrator = CarbonAwareOrchestrator()

    ranked = orchestrator.rank_resources(providers, prediction, request)

    assert ranked is not None
    assert len(ranked) == 1
