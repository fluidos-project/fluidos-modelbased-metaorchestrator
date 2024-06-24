from collections.abc import Callable
from pathlib import Path

import pytest  # type: ignore
from pytest_kubernetes.providers import select_provider_manager  # type: ignore
from pytest_kubernetes.providers.base import AClusterManager  # type: ignore


@pytest.fixture
def tmp_test_path(tmp_path: Path) -> Path:
    """
    This is used mostly during test development to divert any tmp_path
    folder created to a specific location browsable by developper

    Args:
        tmp_path (Path): _description_

    Returns:
        Path: _description_
    """
    return tmp_path

    # Uncomment during local development
    # tmp_test_path = Path(Path(__file__).parent.parent.joinpath("tmp_fluidos"))
    # if tmp_test_path.exists():
    #     shutil.rmtree(tmp_test_path.as_posix())
    # return tmp_test_path


@pytest.fixture
def sample_dataset_path() -> Callable[[str], Path]:
    def run_fixture(dataset_name: str) -> Path:
        return Path(
            Path(__file__).parent,
            f"data_pipeline/dataset_resources/sample_dataset_{dataset_name}",
        )

    return run_fixture


@pytest.fixture(scope="session")
def k8s_with_configuration(request):
    cluster: AClusterManager = select_provider_manager()("test-cluster")

    cluster.start()

    yield cluster

    cluster.delete()
