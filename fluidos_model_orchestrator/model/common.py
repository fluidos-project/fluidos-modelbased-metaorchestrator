import zipfile
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download  # type: ignore

from fluidos_model_orchestrator.common import ModelPredictRequest


class ModelInterface(ABC):

    @abstractmethod
    def get_columns_dependencies(self) -> dict[str, Any]:
        raise NotImplementedError("Not implemented: abstract method")

    @staticmethod
    def create_sample_request() -> ModelPredictRequest:
        raise NotImplementedError("Not implemented: abstract method")


def download_model_from_hf(repo_id: str, model_file_name: str) -> Path:
    downloaded_model_path = hf_hub_download(repo_id=repo_id, filename=model_file_name)

    unzipped_model_path = Path(downloaded_model_path).parent.as_posix()
    with zipfile.ZipFile(downloaded_model_path, 'r') as zip_ref:
        zip_ref.extractall(unzipped_model_path)

    return Path(unzipped_model_path)


MODEL_FILE_NAME = "latest_model_version.zip"
