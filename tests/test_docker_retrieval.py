import pytest  # type: ignore

from fluidos_model_orchestrator.container import _get_image_name_parts
from fluidos_model_orchestrator.container import _retrieve_image
from fluidos_model_orchestrator.container import extract_image_embedding
from fluidos_model_orchestrator.container import ImageData
# from fluidos_model_orchestrator.container import _compute_embedding


def test_docker_image_name_handling() -> None:
    valid_names = [
        ("postgres:latest", "postgres", "latest"),
        ("postgres:16.2", "postgres", "16.2"),
        ("postgres", "postgres", "latest"),
        ("ibmcom/db2", "ibmcom/db2", "latest"),
        ("ibmcom/db2:11.5.0.0", "ibmcom/db2", "11.5.0.0"),
    ]

    for full_name, name, tag in valid_names:
        n, t = _get_image_name_parts(full_name)

        assert n is not None
        assert n == name
        if tag != "latest":
            assert t == tag
        else:
            assert t is None or t == "latest"


def test_docker_image_retrieval() -> None:
    valid_docker_images = [
        "postgres:latest",
        "postgres:16.2",
        "postgres"
    ]

    for image in valid_docker_images:
        image_data = _retrieve_image(image)

        assert image_data is not None, image
        assert type(image_data) is ImageData


@pytest.mark.skip()
def test_embedding_extraction() -> None:
    raise NotImplementedError()


@pytest.mark.skip()
def test_extract_image_embedding() -> None:
    valid_docker_images = [
        "postgres:latest",
        "postgres:16.2"
        "postgres"
    ]

    for image in valid_docker_images:
        assert extract_image_embedding(image) is not None, image

    raise NotImplementedError()
