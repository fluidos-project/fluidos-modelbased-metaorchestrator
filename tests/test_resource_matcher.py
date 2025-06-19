from fluidos_model_orchestrator.common.flavor import Flavor
from fluidos_model_orchestrator.common.flavor import FlavorCharacteristics
from fluidos_model_orchestrator.common.flavor import FlavorK8SliceData
from fluidos_model_orchestrator.common.flavor import FlavorMetadata
from fluidos_model_orchestrator.common.flavor import FlavorSpec
from fluidos_model_orchestrator.common.flavor import FlavorType
from fluidos_model_orchestrator.common.flavor import FlavorTypeData
from fluidos_model_orchestrator.common.flavor import GPUData
from fluidos_model_orchestrator.common.resource import Resource


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


def test_should_match() -> None:
    flavor = Flavor(
        metadata=FlavorMetadata(
            name='fluidos.eu-k8slice-2f1640fd4dfb151b4d81ad9590dcb6cc',
            owner_references={'apiVersion': 'nodecore.fluidos.eu/v1alpha1', 'kind': 'Node', 'name': 'fluidos-consumer-1-worker', 'uid': 'bcbc5b65-3de6-434a-aef9-a329317a189d'}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(
                        cpu='1947481697n',
                        architecture='amd64',
                        gpu=GPUData(cores=0, memory=0, model=''),
                        memory='3735836Ki', pods='110', storage='0'
                    ),
                    policies={'partitionability': {'cpuMin': '0', 'cpuStep': '1', 'gpuMin': '0', 'gpuStep': '0', 'memoryMin': '0', 'memoryStep': '100Mi', 'podsMin': '0', 'podsStep': '0'}},
                    properties={}
                )
            ),
            location={'additionalNotes': 'None', 'city': 'Turin', 'country': 'Italy', 'latitude': '10', 'longitude': '58'},
            network_property_type='networkProperty',
            owner={'domain': 'fluidos.eu', 'ip': '172.18.0.7:30000', 'nodeID': 'ekvjnuvsel'},
            providerID='ekvjnuvsel',
            price={'amount': '', 'currency': '', 'period': ''}
        )
    )

    req = Resource(id="123", architecture="amd64")

    assert req.can_run_on(flavor)
