import kubernetes
import pkg_resources  # type: ignore
import pytest
from pytest_kubernetes.providers.base import AClusterManager

from fluidos_model_orchestrator.configuration import _build_k8s_client
from fluidos_model_orchestrator.resources.rear.remote_resource_provider import RemoteResourceProvider


def test_basic_creation(k8s: AClusterManager) -> None:
    k8s.create()

    myconfig = kubernetes.client.Configuration()
    kubernetes.config.kube_config.load_kube_config(client_configuration=myconfig, config_file=str(k8s.kubeconfig))

    k8s_client = _build_k8s_client(myconfig)

    provider = RemoteResourceProvider(
        "id",
        {
            "domain": "fluidos.eu",
            "ip": "172.18.0.2:30000",
            "nodeID": "l6936ty08l",
        },
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
        {
            "domain": "fluidos.eu",
            "ip": "172.18.0.2:30000",
            "nodeID": "l6936ty08l",
        },
        "peeringcandidate-fluidos.eu-k8s-fluidos-c3978e7c",
        "reservation-test-sample",
        "default",
        k8s_client
    )

    assert provider

    contract_information = provider._buy()

    assert contract_information
