from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

import docker  # type: ignore
from docker.errors import DockerException  # type: ignore
from docker.models.images import Image  # type: ignore

from .common import ContainerImageEmbedding


def extract_image_embedding(image: str) -> ContainerImageEmbedding:
    return ContainerImageEmbedding(
        image=image,
    )
    docker.errors.DockerException
    image_data: ImageData = _retrieve_image(image)
    if image_data.is_valid():
        return ContainerImageEmbedding(
            image=image,
            embedding=_compute_embedding(image_data)
        )
    else:
        return ContainerImageEmbedding(
            image=image,
        )


def _compute_embedding(image_data: ImageData | None) -> str | None:
    if not image_data:
        return None

    digest = sha256(usedforsecurity=False)

    digest.update(image_data.metadata().encode())
    for layer in image_data.layers():
        digest.update(layer.encode())

    return digest.digest().hex()


def _get_image_name_parts(image_name: str) -> tuple[str, str | None]:
    if "@" in image_name:
        # as regitry/namespace/image:tag@sha
        image_name = image_name.split("@")[0]
    if ":" in image_name:
        try:
            path_parts = image_name.split("/")

            [image, tag] = path_parts[-1].split(":")
        except ValueError as e:
            print(image_name)
            raise e

        return ("/".join(path_parts[:-1] + [image]), tag)
    else:
        return image_name, None  # None defaults to latest


def _retrieve_image(image_name: str) -> ImageData:
    image, tag = _get_image_name_parts(image_name)

    try:
        client = docker.from_env()
        data: Image = client.images.pull(image, tag=tag)

        return ImageData(data)
    except DockerException:
        return ImageData()


@dataclass(frozen=True)
class ImageData:
    _image_obj: Image | None = None

    def metadata(self) -> str:
        if self._image_obj is not None:
            return json.dumps(self._image_obj.attrs)
        raise ValueError()

    def layers(self) -> list[str]:
        if self._image_obj is not None:
            return [json.dumps(layer) for layer in self._image_obj.history()]
        raise ValueError()

    def is_valid(self) -> bool:
        return self._image_obj is not None
