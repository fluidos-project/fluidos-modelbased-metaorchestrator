import importlib.resources

from pytest_kubernetes.providers import AClusterManager


def test_service_credentials(k8s: AClusterManager) -> None:
    k8s.create()
    file = importlib.resources.files(__package__).joinpath("examples") / "credentials-secret.yaml"
    k8s.apply(file)
