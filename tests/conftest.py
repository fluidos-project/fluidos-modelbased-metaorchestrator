import pytest
from pytest_kubernetes.providers import AClusterManager
from pytest_kubernetes.providers import select_provider_manager


@pytest.fixture(scope="session")
def k8s_with_configuration(request):
    cluster: AClusterManager = select_provider_manager()("test-cluster")

    cluster.start()

    yield cluster

    cluster.delete()
