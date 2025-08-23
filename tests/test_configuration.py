import importlib.resources
import logging

import kopf  # type: ignore
import kubernetes  # type: ignore
import pytest  # type: ignore
from pytest_kubernetes.providers import AClusterManager  # type: ignore

from fluidos_model_orchestrator.configuration import Configuration
from fluidos_model_orchestrator.configuration import enrich_configuration


def test_failing_missing_config_map(k8s: AClusterManager) -> None:
    k8s.create()
    myconfig = kubernetes.client.Configuration()  # type: ignore
    kubernetes.config.kube_config.load_kube_config(client_configuration=myconfig, config_file=str(k8s.kubeconfig))

    config = Configuration()

    logger = logging.getLogger(__name__)

    with pytest.raises(expected_exception=ValueError):
        enrich_configuration(config, kopf.OperatorSettings(), None, None, {}, logger, myconfig)

    k8s.delete()


def test_check_identity() -> None:
    config = Configuration()
    config.identity["domain"] = "ibm.fluidos.eu"
    config.identity["ip"] = "9.2.3.4:30000"
    config.identity["nodeID"] = "my_amazing_node_ID"

    assert config.check_identity({
        "domain": "ibm.fluidos.eu",
        "ip": "9.2.3.4:30000",
        "nodeID": "my_amazing_node_ID",
    })

    assert not config.check_identity({
        "domain": "ibm.fluidos.eu",
        "ip": "9.2.3.4:30001",
        "nodeID": "another_amazing_node_ID",
    })


def test_configuration_enrichment_with_k8s(k8s: AClusterManager) -> None:
    k8s.create()

    k8s.apply(importlib.resources.files(__package__) / "node/examples/fluidos-network-manager-identity-config-map.yaml")
    k8s.apply(importlib.resources.files(__package__) / "data/example-mbmo-config-map.yaml")

    myconfig = kubernetes.client.Configuration()  # type: ignore
    kubernetes.config.kube_config.load_kube_config(client_configuration=myconfig, config_file=str(k8s.kubeconfig))

    config = Configuration(namespace="default")

    logger = logging.getLogger(__name__)

    enrich_configuration(config, kopf.OperatorSettings(), None, None, {}, logger, myconfig)

    assert config.k8s_client is not None
    assert len(config.identity)
    assert config.identity["domain"] == "ibm.fluidos.eu"
    assert config.identity["ip"] == "9.2.3.4:30000"
    assert config.identity["nodeID"] == "my_amazing_node_ID"
    assert config.api_keys["ELECTRICITY_MAP_API_KEY"] == "TEST_KEY_123!"
    assert config.monitor_contracts is False

    k8s.delete()
