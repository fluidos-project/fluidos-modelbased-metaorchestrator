from collections.abc import Callable
from pathlib import Path

import pytest  # type: ignore
from pytest_kubernetes.providers import select_provider_manager  # type: ignore
from pytest_kubernetes.providers.base import AClusterManager  # type: ignore

from fluidos_model_orchestrator.data_pipeline.data_processor_factory import (
    DataProcessorFactory,
)


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


@pytest.fixture
def create_sample_json_dataset(
    tmp_test_path: Path, sample_dataset_path: Callable[[str], Path], subset: str
) -> Callable[[str, str], Path]:

    def run_fixture(
        dataset_name: str,
        subset: str
    ) -> Path:
        dataset_processor = DataProcessorFactory().create_dataset_processor(
            dataset_name,
            dataset_path=sample_dataset_path(dataset_name),
            parent_output_dataset_path=tmp_test_path.joinpath("json"),
            dataset_version="3",
            pods_number=100,
            mode="evaluation",
            machines_number=3,
        )

        json_preprocessed_dataset_path = dataset_processor.convert_to_json(subset=subset)
        return json_preprocessed_dataset_path

    return run_fixture


@pytest.fixture(scope="session")
def k8s_with_configuration(request):
    cluster: AClusterManager = select_provider_manager()("test-cluster")

    cluster.start()

    yield cluster

    cluster.delete()
