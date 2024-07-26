import logging

import kopf  # type: ignore
import kubernetes  # type: ignore
import pkg_resources  # type: ignore
from pytest_kubernetes.providers import AClusterManager  # type: ignore

from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.configuration import _build_k8s_client
from fluidos_model_orchestrator.configuration import Configuration
from fluidos_model_orchestrator.configuration import enrich_configuration
from fluidos_model_orchestrator.resources import REARResourceFinder


def test_find_local_no_nodes(k8s: AClusterManager) -> None:
    k8s.create()

    myconfig = kubernetes.client.Configuration()
    kubernetes.config.kube_config.load_kube_config(client_configuration=myconfig, config_file=str(k8s.kubeconfig))

    configuration = Configuration(
        k8s_client=_build_k8s_client(myconfig)
    )

    finder = REARResourceFinder(configuration)

    res_providers = finder._find_local(Resource(id="123", architecture="amd64"), "default")

    assert len(res_providers) == 0


def test_solver_creation_and_check(k8s: AClusterManager) -> None:
    k8s.create()

    k8s.apply(pkg_resources.resource_filename(__name__, "node/crds/nodecore.fluidos.eu_solvers.yaml"))
    k8s.apply(pkg_resources.resource_filename(__name__, "node/examples/fluidos-network-manager-identity-config-map.yaml"))
    k8s.apply(pkg_resources.resource_filename(__name__, "data/example-mbmo-config-map.yaml"))

    myconfig = kubernetes.client.Configuration()
    kubernetes.config.kube_config.load_kube_config(client_configuration=myconfig, config_file=str(k8s.kubeconfig))

    configuration = Configuration()

    logger = logging.getLogger(__name__)

    enrich_configuration(configuration, kopf.OperatorSettings(), None, None, {}, logger, myconfig)

    finder = REARResourceFinder(configuration)

    body, _ = finder._resource_to_solver_request(
        Resource(id="123", architecture="amd64"),
        "intent-123"
    )

    assert len(k8s.kubectl(["get", "solver"])["items"]) == 0

    assert finder._check_solver_status("foo", "default") is None

    solver_id = finder._initiate_search(body, "default")

    assert solver_id is not None

    solvers = k8s.kubectl(["get", "solver"])["items"]

    assert len(solvers) == 1

    solver = finder._check_solver_status(solver_id, "default")

    assert solver is not None
    assert solver["metadata"]["name"] == solver_id


def test_retrieve_peering_candidate_list(k8s: AClusterManager) -> None:
    k8s.create()

    k8s.apply(pkg_resources.resource_filename(__name__, "node/crds/advertisement.fluidos.eu_discoveries.yaml"))
    k8s.apply(pkg_resources.resource_filename(__name__, "node/examples/fluidos-network-manager-identity-config-map.yaml"))
    k8s.apply(pkg_resources.resource_filename(__name__, "data/example-mbmo-config-map.yaml"))

    k8s.apply(pkg_resources.resource_filename(__name__, "node/examples/nginx-w-intent-discovery.yaml"))

    k8s.kubectl(["patch", "discovery", "discovery-nginx-w-intent-solver",
                 "--patch-file", pkg_resources.resource_filename(__name__, "node/examples/nginx-w-intent-discovery-patch.yaml"),
                 "--type", "merge", "--subresource", "status"])

    res = k8s.kubectl(["get", "discovery/discovery-nginx-w-intent-solver"])

    assert "status" in res

    myconfig = kubernetes.client.Configuration()
    kubernetes.config.kube_config.load_kube_config(client_configuration=myconfig, config_file=str(k8s.kubeconfig))

    configuration = Configuration()

    logger = logging.getLogger(__name__)

    enrich_configuration(configuration, kopf.OperatorSettings(), None, None, {}, logger, myconfig)

    finder = REARResourceFinder(configuration)

    solver_id = "nginx-w-intent-solver"

    candidates = finder._retrieve_peering_candidates(solver_id, "default")

    assert candidates is not None
    assert len(candidates) == 1

    k8s.delete()


def test_temporary_flavor_update(k8s: AClusterManager) -> None:
    k8s.create()

    k8s.apply(pkg_resources.resource_filename(__name__, "node/examples/fluidos-network-manager-identity-config-map.yaml"))
    k8s.apply(pkg_resources.resource_filename(__name__, "data/example-mbmo-config-map.yaml"))
    k8s.apply(pkg_resources.resource_filename(__name__, "node/crds/nodecore.fluidos.eu_flavours.yaml"))
    k8s.apply(pkg_resources.resource_filename(__name__, "node/examples/example-flavor.yaml"))

    myconfig = kubernetes.client.Configuration()
    kubernetes.config.kube_config.load_kube_config(client_configuration=myconfig, config_file=str(k8s.kubeconfig))

    configuration = Configuration()
    logger = logging.getLogger(__name__)

    enrich_configuration(configuration, kopf.OperatorSettings(), None, None, {}, logger, myconfig)

    finder = REARResourceFinder(configuration)

    flavors = finder._get_locally_available_flavors("default")

    assert len(flavors) == 1

    flavor = flavors[0]

    finder.update_local_flavor(flavor, {"carbon": ["test", "123"]}, "default")

    after_flavors = finder._get_locally_available_flavors("default")

    assert len(after_flavors) == 1

    assert "carbon" in after_flavors[0].optional_fields

    k8s.delete()
