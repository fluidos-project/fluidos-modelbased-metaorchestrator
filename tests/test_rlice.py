import pkg_resources
import yaml

from fluidos_model_orchestrator.model import convert_to_model_request
from fluidos_model_orchestrator.model import get_model_object
from fluidos_model_orchestrator.common import Flavor
from fluidos_model_orchestrator.common import FlavorCharacteristics
from fluidos_model_orchestrator.common import FlavorK8SliceData
from fluidos_model_orchestrator.common import FlavorMetadata
from fluidos_model_orchestrator.common import FlavorSpec
from fluidos_model_orchestrator.common import FlavorType
from fluidos_model_orchestrator.common import FlavorTypeData
from fluidos_model_orchestrator.common import GPUData
from fluidos_model_orchestrator.common import Intent
from fluidos_model_orchestrator.common import KnownIntent
from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.resources.rear.local_resource_provider import LocalResourceProvider
from fluidos_model_orchestrator.model.rlice.model import RliceOrchestrator

def test_validate_provider_characteristics() -> None:

    bad_no_price = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="2000n", memory="10Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={},
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={
            }
        )
    ))

    good = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="2000n", memory="10Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={},
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={
                "amount":"0.16",
                "currency":"USD",
                "period":"hourly"
            }
        )
    ))

    providers: list[ResourceProvider] = [bad_no_price,good]

    with pkg_resources.resource_stream(__name__, "examples/test-pod-w-intent.yaml") as stream:
        request = convert_to_model_request(
            yaml.safe_load(stream)["spec"],
            "fluidos"
        )
        print(request)
    prediction = ModelPredictResponse(
        "resp-123",
        Resource(id="res-123", architecture="amd64", cpu="250m", memory="64Mi"),
        delay=0
    )
    orchestrator = RliceOrchestrator()

    ranked = orchestrator.rank_resources(providers, prediction, request)

    #assert ranked is not None
    #assert len(ranked) == 1

test_validate_provider_characteristics()