from fluidos_model_orchestrator.common import Flavor
from fluidos_model_orchestrator.common import FlavorCharacteristics
from fluidos_model_orchestrator.common import FlavorK8SliceData
from fluidos_model_orchestrator.common import FlavorMetadata
from fluidos_model_orchestrator.common import FlavorSpec
from fluidos_model_orchestrator.common import FlavorType
from fluidos_model_orchestrator.common import FlavorTypeData
from fluidos_model_orchestrator.common import GPUData
from fluidos_model_orchestrator.common import Intent
from fluidos_model_orchestrator.resources.rear.local_resource_provider import LocalResourceProvider

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