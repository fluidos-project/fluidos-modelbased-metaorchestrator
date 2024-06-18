from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

import docker
from docker.models.images import Image

from .common import ContainerImageEmbedding


def _extract_image_embedding(image: str) -> ContainerImageEmbedding:
    return ContainerImageEmbedding(
        image=_compute_embedding(_retrieve_image(image))
    )


def _compute_embedding(image_data: ImageData) -> str:
    digest = sha256(usedforsecurity=False)

    digest.update(image_data.metadata().encode())
    for layer in image_data.layers():
        digest.update(layer.encode())

    return digest.digest().hex()


def _get_image_name_parts(image_name: str) -> tuple[str, str | None]:
    if ":" in image_name:
        [a, b] = image_name.split(":")
        return (a, b)
    else:
        return image_name, None  # None defaults to latest


def _retrieve_image(image_name: str) -> ImageData:
    image, tag = _get_image_name_parts(image_name)

    client = docker.from_env()

    data: Image = client.images.pull(image, tag=tag)

    return ImageData(data)


@dataclass(frozen=True)
class ImageData:
    _image_obj: Image

    def metadata(self) -> str:
        return json.dumps(self._image_obj.attrs)

    def layers(self) -> list[str]:
        return [json.dumps(layer) for layer in self._image_obj.history()]
