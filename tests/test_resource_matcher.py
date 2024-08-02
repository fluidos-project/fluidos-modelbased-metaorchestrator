from fluidos_model_orchestrator.common import Flavor
from fluidos_model_orchestrator.common import FlavorCharacteristics
from fluidos_model_orchestrator.common import FlavorK8SliceData
from fluidos_model_orchestrator.common import FlavorMetadata
from fluidos_model_orchestrator.common import FlavorSpec
from fluidos_model_orchestrator.common import FlavorType
from fluidos_model_orchestrator.common import FlavorTypeData
from fluidos_model_orchestrator.common import GPUData
from fluidos_model_orchestrator.common import Resource


def test_request_satisfied():
    res = Resource(id="foo", cpu="2n", memory="10Mi")

    flavor = Flavor(
        metadata=FlavorMetadata(name="bar", owner_references={}),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(
                        architecture="amd64",
                        cpu="2000000n",
                        memory="100Gi",
                        gpu=GPUData()),
                    policies={},
                    properties={},
                ),
            ),
            location={},
            network_property_type="something",
            providerID="foo",
            owner={},
            price={}
        )
    )

    assert res.can_run_on(flavor)


def test_request_not_sastisfied():
    res = Resource(id="foo", cpu="2n", memory="10Mi", gpu="1")

    metadata = FlavorMetadata(name="bar", owner_references={})

    assert not res.can_run_on(Flavor(
        metadata=metadata,
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(
                        architecture="amd64",
                        cpu="2000000n",
                        memory="100Gi",
                        gpu=GPUData()),
                    policies={},
                    properties={},
                ),
            ),
            location={},
            network_property_type="something",
            providerID="foo",
            owner={},
            price={}
        ))), "Missing GPU"

    assert not res.can_run_on(Flavor(
        metadata=metadata,
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(
                        architecture="amd64",
                        cpu="1n",
                        memory="100Gi",
                        gpu=GPUData(cores=1)),
                    policies={},
                    properties={},
                ),
            ),
            location={},
            network_property_type="something",
            providerID="foo",
            owner={},
            price={}
        ))), "Not enough CPU"
