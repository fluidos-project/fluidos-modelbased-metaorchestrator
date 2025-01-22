from typing import Any

import numpy as np  # type: ignore
import pandas as pd
import torch  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
from torch.utils.data import Dataset  # type: ignore

from fluidos_model_orchestrator.model.utils import FLUIDOS_COL_NAMES


class RankerDataset(Dataset[Any]):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe
        # pod_columns = POD_TEMPLATE_RESOURCE_DF_VALUES[FLUIDOS_DATASETS.GCT]

        #treating pod_id as Categorical Strings
        # pod_id_encoded_data = LabelEncoder().fit_transform(pod_filename_list)
        pod_filename_list = self.data[FLUIDOS_COL_NAMES.POD_FILE_NAME].values
        self.label_encoders = {}
        self.label_encoders["pod_filename"] = LabelEncoder()
        self.label_encoders["pod_filename"].fit(pod_filename_list)

        # pod manifest
        pod_manifest_list = self.data[FLUIDOS_COL_NAMES.POD_MANIFEST].values
        self.sentence_transformer_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

        pod_cpu_list = self.data[FLUIDOS_COL_NAMES.POD_CPU].values

        pod_memory_list = self.data[FLUIDOS_COL_NAMES.POD_MEMORY].values

        pod_location_list = self.data[FLUIDOS_COL_NAMES.POD_LOCATION].values
        # pod_location_encoded_data = LabelEncoder().fit_transform(pod_location_list)
        self.label_encoders["pod_location"] = LabelEncoder()
        self.label_encoders["pod_location"].fit(pod_location_list)

        #treating template_resource_id as Categorical Strings
        template_resource_id_list = self.data[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_ID].values
        # template_resource_id_encoded_data = LabelEncoder().fit_transform(template_resource_id)
        self.label_encoders["template_resource_id"] = LabelEncoder()
        self.label_encoders["template_resource_id"].fit(template_resource_id_list)

        template_resoruce_cpu_list = self.data[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU].values
        template_resource_memory_list = self.data[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY].values

        template_location_list = self.data[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_LOCATION].values
        # template_location_encoded_data = LabelEncoder().fit_transform(template_location)
        self.label_encoders["template_location"] = LabelEncoder()
        self.label_encoders["template_location"].fit(template_location_list)

        self.pod_id, self.pod_manifest_embeddings, self.pod_cpu, self.pod_mem, self.pod_location, self.template_resource_id, self.template_cpu, self.template_mem, self.template_location = self.pre_process_inputs(pod_filename_list, pod_manifest_list, pod_cpu_list, pod_memory_list, pod_location_list, template_resource_id_list, template_resoruce_cpu_list, template_resource_memory_list, template_location_list)

        self.unique_pod_ids = np.array(list(set(self.pod_id)))
        self.unique_pod_locations = np.array(list(set(self.pod_location)))
        self.unique_template_resource_ids = np.array(list(set(self.template_resource_id)))
        self.unique_template_locations = np.array(list(set(self.template_location)))

        # TODO ['pod_location', 'template_resource_location']

        if FLUIDOS_COL_NAMES.TARGET_PERFORMANCE_RESOURCES_AUGMENTATION_COL in self.data:
            self.target_performance = torch.tensor(self.data[FLUIDOS_COL_NAMES.TARGET_PERFORMANCE_RESOURCES_AUGMENTATION_COL].values, dtype=torch.float32)

    def pre_process_inputs(self, pod_filename_list: Any, pod_manifest_list: Any, pod_cpu_list: Any, pod_memory_list: Any, pod_location_list: Any, template_resource_id_list: Any, template_resoruce_cpu_list: Any, template_resource_memory_list: Any, template_location_list: Any) -> Any:
        pod_id_encoded_data = self.label_encoders["pod_filename"].transform(pod_filename_list)
        pod_id = torch.tensor(pod_id_encoded_data, dtype=torch.long)

        # Generate embeddings for each entry in pod_manifest
        pod_manifest_embeddings = self.sentence_transformer_model.encode(pod_manifest_list, convert_to_tensor=True)

        #Pod_CPU
        pod_cpu = torch.tensor(pod_cpu_list, dtype=torch.float32)

        pod_mem = torch.tensor(pod_memory_list, dtype=torch.float32)

        pod_location_encoded_data = self.label_encoders["pod_location"].transform(pod_location_list)
        pod_location = torch.tensor(pod_location_encoded_data, dtype=torch.long)

        template_resource_id_encoded_data = self.label_encoders["template_resource_id"].transform(template_resource_id_list)
        template_resource_id = torch.tensor(template_resource_id_encoded_data, dtype=torch.long)

        template_cpu = torch.tensor(template_resoruce_cpu_list, dtype=torch.float32)

        template_mem = torch.tensor(template_resource_memory_list, dtype=torch.float32)

        template_location_encoded_data = self.label_encoders["template_location"].transform(template_location_list)
        template_location = torch.tensor(template_location_encoded_data, dtype=torch.long)

        return pod_id, pod_manifest_embeddings, pod_cpu, pod_mem, pod_location, template_resource_id, template_cpu, template_mem, template_location

    def encode_sentence(self, sentence: Any, vocab: dict[str, Any]) -> list[Any]:
        return [vocab.get(token, vocab["<unk>"]) for token in self.tokenize(sentence)]

    def build_vocab(self, data: Any) -> dict[str, int]:
        tokenized_data = [self.tokenize(sentence) for sentence in data]
        vocab = {"<pad>": 0, "<unk>": 1}  # Special tokens
        index = len(vocab)

        for sentence in tokenized_data:
            for token in sentence:
                if token not in vocab:
                    vocab[token] = index
                    index += 1
        return vocab

    def tokenize(self, text: Any) -> list[str]:
        return str(text).lower().split()

    def __len__(self) -> int:
        return len(self.pod_id)

    def __getitem__(self, idx: Any) -> tuple[tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any], torch.Tensor]:
        # features = torch.tensor(self.features[idx], dtype=torch.float32)
        # label = torch.tensor(self.labels[idx], dtype=torch.long)
        # self.pod_embeddings[index]
        return (self.pod_id[idx], self.pod_cpu[idx], self.pod_mem[idx], self.pod_location[idx], self.pod_manifest_embeddings[idx], self.template_resource_id[idx], self.template_cpu[idx], self.template_mem[idx], self.template_location[idx]), self.target_performance[idx]


def get_tensor_input(pods_assigment_df: Any, dataset: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    pod_filename_list = pods_assigment_df[FLUIDOS_COL_NAMES.POD_FILE_NAME].values
    pod_manifest_list = pods_assigment_df[FLUIDOS_COL_NAMES.POD_MANIFEST].values
    pod_cpu_list = pods_assigment_df[FLUIDOS_COL_NAMES.POD_CPU].values
    pod_memory_list = pods_assigment_df[FLUIDOS_COL_NAMES.POD_MEMORY].values
    pod_location_list = pods_assigment_df[FLUIDOS_COL_NAMES.POD_LOCATION].values
    template_resource_id_list = pods_assigment_df[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_ID].values
    template_resoruce_cpu_list = pods_assigment_df[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU].values
    template_resource_memory_list = pods_assigment_df[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY].values
    template_location_list = pods_assigment_df[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_LOCATION].values

    pod_id, pod_manifest_embeddings, pod_cpu, pod_mem, pod_location, template_resource_id, template_cpu, template_mem, template_location = dataset.pre_process_inputs(
        pod_filename_list,
        pod_manifest_list,
        pod_cpu_list,
        pod_memory_list,
        pod_location_list,
        template_resource_id_list,
        template_resoruce_cpu_list,
        template_resource_memory_list,
        template_location_list
    )

    input_sample_batch = (pod_id, pod_cpu, pod_mem, pod_location, pod_manifest_embeddings, template_resource_id, template_cpu, template_mem, template_location)
    return input_sample_batch
