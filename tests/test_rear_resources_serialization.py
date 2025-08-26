import kubernetes  # type: ignore
from pytest_kubernetes.providers import AClusterManager  # type: ignore

from fluidos_model_orchestrator.common.flavor import Flavor
from fluidos_model_orchestrator.common.flavor import FlavorCharacteristics
from fluidos_model_orchestrator.common.flavor import FlavorK8SliceData
from fluidos_model_orchestrator.common.flavor import FlavorMetadata
from fluidos_model_orchestrator.common.flavor import FlavorSpec
from fluidos_model_orchestrator.common.flavor import FlavorType
from fluidos_model_orchestrator.common.flavor import FlavorTypeData
from fluidos_model_orchestrator.configuration import _build_k8s_client
from fluidos_model_orchestrator.resources.rear.local_resource_provider import LocalResourceProvider
from fluidos_model_orchestrator.resources.rear.remote_resource_provider import RemoteResourceProvider


def test_local() -> None:
    provider = LocalResourceProvider(
        flavor=Flavor(
            metadata=FlavorMetadata(name="test", owner_references={}),
            spec=FlavorSpec(
                availability=True,
                flavor_type=FlavorTypeData(
                    type_identifier=FlavorType.K8SLICE,
                    type_data=FlavorK8SliceData(
                        characteristics=FlavorCharacteristics(
                            cpu="",
                            architecture="",
                            memory="",
                        )
                    ),
                ),
                network_property_type="",
                owner={
                    "nodeID": "my.nodeID",
                    "domain": "my.domain.com"
                },
                providerID="foo",
            )
        ),
        id="foo"
    )

    assert provider
    assert str(provider) == "LocalResourceProvider[test{my.nodeID@my.domain.com}]"


def test_remote(k8s: AClusterManager) -> None:
    k8s.create()

    myconfig = kubernetes.client.Configuration()  # type: ignore
    kubernetes.config.kube_config.load_kube_config(client_configuration=myconfig, config_file=str(k8s.kubeconfig))

    k8s_client = kubernetes.client.CustomObjectsApi(_build_k8s_client(myconfig))

    provider = RemoteResourceProvider(
        id="123",
        flavor=Flavor(
            metadata=FlavorMetadata(name="test", owner_references={}),
            spec=FlavorSpec(
                availability=True,
                flavor_type=FlavorTypeData(
                    type_identifier=FlavorType.K8SLICE,
                    type_data=FlavorK8SliceData(
                        characteristics=FlavorCharacteristics(
                            cpu="",
                            architecture="",
                            memory="",
                        )
                    ),
                ),

                network_property_type="",
                owner={
                    "nodeID": "my.nodeID",
                    "domain": "my.domain.com"
                },
                providerID="foo",
            )
        ),
        api_client=k8s_client,
        peering_candidate="peering_candidate",
        reservation="reservation",
        seller={
            "nodeID": "my.nodeID",
            "domain": "my.domain.com"
        }
    )

    assert provider
    assert str(provider) == "RemoteResourceProvider[test{my.nodeID@my.domain.com}]"
