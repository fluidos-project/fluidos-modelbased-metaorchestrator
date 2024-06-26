from typing import Any

import kubernetes  # type: ignore
import pkg_resources  # type: ignore
import pytest  # type: ignore
from pytest_kubernetes.providers.base import AClusterManager  # type: ignore

from fluidos_model_orchestrator.common import Flavor
from fluidos_model_orchestrator.common import FlavorCharacteristics
from fluidos_model_orchestrator.common import FlavorType
from fluidos_model_orchestrator.configuration import _build_k8s_client
from fluidos_model_orchestrator.resources.rear.remote_resource_provider import RemoteResourceProvider


def test_basic_creation(k8s: AClusterManager) -> None:
    k8s.create()

    myconfig = kubernetes.client.Configuration()
    kubernetes.config.kube_config.load_kube_config(client_configuration=myconfig, config_file=str(k8s.kubeconfig))

    k8s_client = _build_k8s_client(myconfig)

    owner: dict[str, Any] = {
        "domain": "fluidos.eu",
        "ip": "172.18.0.2:30000",
        "nodeID": "l6936ty08l",
    }

    provider = RemoteResourceProvider(
        "id",
        Flavor(
            "flavor_id",
            FlavorType.K8SLICE,
            FlavorCharacteristics("1", "amd", "0", "1000"),
            owner,
            providerID="foo"
        ),
        "peeringcandidate-fluidos.eu-k8s-fluidos-c3978e7c",
        "reservation-test-sample",
        "default",
        k8s_client
    )

    assert provider


@pytest.mark.skip()
def test_resource_buying(k8s: AClusterManager) -> None:
    k8s.create()

    myconfig = kubernetes.client.Configuration()
    kubernetes.config.kube_config.load_kube_config(client_configuration=myconfig, config_file=str(k8s.kubeconfig))

    k8s_client = _build_k8s_client(myconfig)

    k8s.apply(pkg_resources.resource_filename(__name__, "node/crds/tests/node/crds/reservation.fluidos.eu_reservations.yaml"))

    # create reservation
    k8s.apply(pkg_resources.resource_filename(__name__, "node/crds/tests/node/examples/example-reservation-test.yaml"))

    provider = RemoteResourceProvider(
        "id",
        Flavor(
            "flavor_id",
            FlavorType.K8SLICE,
            FlavorCharacteristics("1", "amd", "0", "1000"),
            {
                "domain": "fluidos.eu",
                "ip": "172.18.0.2:30000",
                "nodeID": "l6936ty08l",
            },
            providerID="foo"
        ),
        "peeringcandidate-fluidos.eu-k8s-fluidos-c3978e7c",
        "reservation-test-sample",
        "default",
        k8s_client
    )

    assert provider

    contract_information = provider._buy()

    assert contract_information
