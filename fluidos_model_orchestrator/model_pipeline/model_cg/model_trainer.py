import ast
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore
import numpy.typing as npt  # type: ignore
import pandas as pd
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim.lr_scheduler as lr_scheduler  # type: ignore
import tqdm
from sentence_transformers import SentenceTransformer  # type: ignore
from torch.nn.utils.rnn import pad_sequence  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from torch.utils.data import Dataset

from fluidos_model_orchestrator.data_pipeline.data_util import FLUIDOS_COL_NAMES
from fluidos_model_orchestrator.data_pipeline.data_util import PIPELINE_FILES
from fluidos_model_orchestrator.model.model_cg.model import OrchestrationModel
from fluidos_model_orchestrator.model.utils import MODEL_HF_NAMES
from fluidos_model_orchestrator.model.utils import MODEL_HF_VERSION
from fluidos_model_orchestrator.model.utils import MODEL_TYPES
from fluidos_model_orchestrator.model_pipeline.model_trainer import BaseModelTrainer
from fluidos_model_orchestrator.model_pipeline.model_util import MODEL_CHECKPOINT_NAMES
from fluidos_model_orchestrator.model_pipeline.model_util import TRAINIG_LOGS_NAMES
# from sklearn.model_selection import train_test_split   # type: ignore
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def calculate_accuracy(target: torch.Tensor, logits: torch.Tensor, batch_size: int) -> float:
    accuracy = sum(target.argmax(dim=1).detach().cpu().numpy() ==
                   logits.argmax(dim=1).detach().cpu().numpy()) / batch_size
    return accuracy


def fill_target_logits(logits: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    target_logits = torch.zeros_like(logits)
    for i in range(target_logits.shape[0]):
        target_logits[i, target_tensor[i].long()] = 1.
    return target_logits


class CGModelTrainer(BaseModelTrainer):
    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        target_column: str,
        epochs: int = 15,
        learning_rate: float = 1e-4,
        dataset_version: str = "v0",
        dataset_size: int = 100,
        dataset_max_size: int = -1,
        train_ratio: float = 0.9,
        subset: str | None = None,
        mode: str = "train",
        batch_size: int = 8,
        device: str = "cpu",
        lr_milestones: list[int] = [3, 100],
        validation_freq: int = 5,
        embedding_batch_size: int = 128,
        pod_embedding_size: int = 512,
        embedding_size: int = 8,
        fc1_size: int = 1024,
        fc2_size: int = 512,
        fc3_size: int = 512,
        dropout1: float = 0.7,
        dropout2: float = 0.35,
        dropout3: float = 0.35,
        aggregation_mode: str = 'mean',
        load_from_generated: bool = False
    ) -> None:
        super().__init__(MODEL_TYPES.CG,
                         dataset_path,
                         output_dir,
                         epochs=epochs,
                         target_column=target_column,
                         dataset_max_size=dataset_max_size,
                         validation_freq=validation_freq,
                         repo_id="DUMMY_REPO_ID",
                         learning_rate=learning_rate)
        self.learning_rate = learning_rate
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.lr_milestones = lr_milestones
        self.validation_freq = validation_freq
        self.train_ratio = train_ratio
        self.dataset_path = dataset_path
        self.load_from_generated = load_from_generated
        self.checkpoint_path = self.checkpoint_dir.joinpath(f"{MODEL_CHECKPOINT_NAMES.CG_INTERMEDIATE}_{epochs-1}.pt")

        self.model_config: dict[str, Any] = {
            "fc1_size": fc1_size,
            "fc2_size": fc2_size,
            "fc3_size": fc3_size,
            "dropout1": dropout1,
            "dropout2": dropout2,
            "dropout3": dropout3,
            "aggregation_mode": aggregation_mode,
            "pod_embedding_size": pod_embedding_size,
            "tr_embedding_size": embedding_size,
        }

        self.dataset_config: dict[str, Any] = {
            "dataset_path": self.dataset_path.as_posix(),
            "output_dir": output_dir.as_posix(),
            "dataset_size": dataset_size,
            "dataset_version": dataset_version,
            "embeddings_batch_size": embedding_batch_size,
            "subset": subset,
            "load_from_generated": load_from_generated,
        }

        self.train_loss_over_epochs: list[float] = []
        self.val_loss_over_epochs: list[float] = []
        self.train_acc_over_epochs: list[float] = []
        self.val_acc_over_epochs: list[float] = []

    def _prepare_dataset_model_specific(self, test_mode: bool = False, training_size_ratio: float = 0.90,
                                        model_tag: str = "model_source", dataset_type: str = "train") -> tuple[Any, Any]:
        if test_mode:
            self.checkpoint_path = self.checkpoint_dir.joinpath(f"{MODEL_CHECKPOINT_NAMES.CG_INTERMEDIATE}_{self.epochs-1}.pt")  # num epochs was changed
        with open(self.dataset_path.joinpath(PIPELINE_FILES.TEMPLATE_RESOURCES_TO_CLASS_ID)) as f:
            self.template_resource2id = json.load(f)
        self.model_config["num_configs"] = len(self.template_resource2id)

        if self.load_from_generated:
            dataset: CGDataset = torch.load(self.checkpoint_dir.parent.parent.joinpath(f"{model_tag}_{dataset_type}.ptd"))  # TODO add model folder (?)
        else:
            subset = self.pods_assigment_df
            dataset_length = (len(subset) // self.batch_size) * self.batch_size
            print("Subset shape", dataset_length)
            if dataset_length == 0:
                raise Exception(f"Length of {dataset_type} subset is too small {len(subset)} to fit the batch size {self.batch_size}")

            subset = subset[:dataset_length]

            dataset = CGDataset(subset,
                                emb_batch_size=self.dataset_config["embeddings_batch_size"])
        torch.save(dataset, self.checkpoint_dir.parent.parent.joinpath(f"{model_tag}_{dataset_type}.ptd"))
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        if dataset_type == "train":
            self.train_dataloader = data_loader
            self.train_size: int = (len(self.train_dataloader.dataset)) // self.batch_size
        else:
            self.val_dataloader = data_loader
            self.val_size: int = (len(self.val_dataloader.dataset)) // self.batch_size

        return data_loader, None

    def _export_hyper_params_model_specific(self) -> dict[str, Any]:
        hyper_params: dict[str, Any] = {
            "train_ratio": self.train_ratio,
            "lr_milestones": str(self.lr_milestones)}
        for key, value in self.model_config.items():
            hyper_params[key] = value
        for key, value in self.dataset_config.items():
            hyper_params[key] = value
        hyper_params['template_resource2id'] = self.template_resource2id
        return hyper_params

    def _evaluate_model_specific(self) -> dict[str, float]:
        evaluation_results = {"total_loss": self.val_loss_over_epochs[-1],
                              "accuracy": self.val_acc_over_epochs[-1]}
        return evaluation_results

    def _build_model_specific(self) -> OrchestrationModel:
        orchestrator = OrchestrationModel(self.model_config, target_column=self.target_column)
        self.orchestrator = orchestrator.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(orchestrator.parameters(),
                                          lr=self.learning_rate)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                  milestones=[milestone for milestone in self.lr_milestones],
                                                  gamma=0.1)
        return self.orchestrator

    def _train_model_specific(self) -> dict[str, list[float]]:
        min_val_loss = 1e10
        print("Training...")
        for epoch in range(self.epochs):
            train_loss, train_accuracy = self.__training(epoch)

            self.scheduler.step()
            self.orchestrator.eval()

            self.train_loss_over_epochs.append(train_loss)
            self.train_acc_over_epochs.append(train_accuracy)

            print(f"Epoch {epoch}, train loss {round(self.train_loss_over_epochs[-1], 4)}",
                  f"Accuracy {round(train_accuracy, 2)}",
                  f"LR: {self.scheduler.get_last_lr()}")
            # self.__save_intetmediate_results(train_loss, 0, epoch)
            # if epoch % self.validation_freq:

            current_val_loss, val_accuracy = self.__validation(epoch)

            self.val_loss_over_epochs.append(current_val_loss)
            self.val_acc_over_epochs.append(val_accuracy)

            print(f"Epoch {epoch}, val loss {round(self.val_loss_over_epochs[-1], 4)}",
                  f"Accuracy {round(val_accuracy, 2)}")

            if current_val_loss < min_val_loss:
                min_val_loss = current_val_loss
                print(f"Current minimal val loss is {min_val_loss}.")
                # print(f"Current minimal val loss is {min_val_loss}. Checkpoint is saved!")
            self.__save_intetmediate_results(train_loss, current_val_loss, epoch)
        self.__save_trainig_logs()
        training_history: dict[str, list[float]] = {"val_loss": self.val_loss_over_epochs,
                                                    "train_loss": self.train_loss_over_epochs,
                                                    "val_accuracy": self.val_acc_over_epochs,
                                                    "train_accuracy": self.train_acc_over_epochs}
        return training_history

    def __training(self, epoch: int) -> tuple[float, float]:
        train_loss: float = 0.0
        train_accuracy: float = 0.0
        with tqdm.tqdm(self.train_dataloader, unit="batch", desc="Epoch") as tepoch:
            k = 0
            for input, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                self.optimizer.zero_grad()  # TODO check if neccessary
                input = [item.to(self.device) for item in input]

                logits = self.orchestrator.forward(input)
                target_logits = fill_target_logits(logits, target)
                loss = self.loss_fn(logits, target_logits)
                loss.backward()
                self.optimizer.step()

                accuracy = calculate_accuracy(target_logits, logits, self.batch_size)
                train_accuracy += accuracy
                k += 1
                train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item(), accuracy=round(accuracy, 2))

        train_accuracy = train_accuracy / self.train_size
        train_loss = train_loss / self.train_size
        return train_loss, train_accuracy

    def __validation(self, epoch: int) -> tuple[float, float]:

        valid_loss: float = 0.0
        val_accuracy: float = 0.0
        with tqdm.tqdm(self.val_dataloader, unit="batch", desc="(val) Epoch") as vepoch:
            for (input, target) in self.val_dataloader:
                vepoch.set_description(f"(val) Epoch {epoch}")
                input = [item.to(self.device) for item in input]
                logits = self.orchestrator.forward(input)
                target_logits = fill_target_logits(logits, target)
                vloss = self.loss_fn(logits, target_logits)
                accuracy = calculate_accuracy(target_logits, logits, self.batch_size)
                val_accuracy += accuracy

                valid_loss += vloss.item()
                vepoch.set_postfix(val_loss=vloss.item(), v_accuracy=round(accuracy, 2))

        valid_loss = valid_loss / self.val_size
        val_accuracy = val_accuracy / self.val_size

        return valid_loss, val_accuracy

    def _predict_model_specific(self, input_dict: dict[str, list[Any]]) -> npt.NDArray[np.float32]:
        return np.zeros(1, dtype=np.float32)

    def _export_model_specific_hyper_params(self) -> dict[str, Any]:
        return self.model_config

    def _load_model_specific(self, model_path: Path | str, load_from_hugging_face: bool = False, load_from_checkpoint: bool = False) -> Any:

        if load_from_hugging_face:
            return OrchestrationModel.from_pretrained(model_path)
        try:
            if load_from_checkpoint:
                self.orchestrator = OrchestrationModel(self.model_config, target_column=self.target_column)
                self.orchestrator.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])  # type: ignore
                self.orchestrator.eval()

                self.loss_fn = nn.CrossEntropyLoss()
                self.optimizer = torch.optim.Adam(self.orchestrator.parameters(),
                                                  lr=self.learning_rate)
                self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                          milestones=[milestone for milestone in self.lr_milestones],
                                                          gamma=0.1)
                return self.orchestrator
            else:
                self.orchestrator = OrchestrationModel(self.model_config, target_column=self.target_column)
                self.orchestrator.load_state_dict(torch.load(model_path.joinpath(MODEL_CHECKPOINT_NAMES.CG_LAST), map_location=self.device)['model_state_dict'])  # type: ignore
                self.orchestrator.eval()

                self.loss_fn = nn.CrossEntropyLoss()
                self.optimizer = torch.optim.Adam(self.orchestrator.parameters(),
                                                  lr=self.learning_rate)
                self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                          milestones=[milestone for milestone in self.lr_milestones],
                                                          gamma=0.1)
                return self.orchestrator  # type: ignore

        except ValueError:
            raise ValueError()

    def __save_trainig_logs(self, ) -> None:
        with open(self.model_dir.joinpath(TRAINIG_LOGS_NAMES.CG_TRAIN_LOG).as_posix(), "w") as f:
            writer = csv.writer(f)
            header = ["Train loss", "Train accuracy"]
            writer.writerow(header)
            rows = [[loss, acc] for loss, acc in zip(self.train_loss_over_epochs, self.train_acc_over_epochs)]

        with open(self.model_dir.joinpath(TRAINIG_LOGS_NAMES.CG_VAL_LOG).as_posix(), "w") as f:
            writer = csv.writer(f)
            header = ["Val loss", "Val accuracy"]
            writer.writerow(header)
            rows = [[loss, acc] for loss, acc in zip(self.val_loss_over_epochs, self.val_acc_over_epochs)]
            writer.writerows(rows)

    def _save_model_specific(self, upload_to_hugging_face: bool = False) -> Path | str | None:
        if upload_to_hugging_face:
            version = MODEL_HF_VERSION.LATEST
            self.orchestrator.push_to_hub(f"fluidos/{MODEL_HF_NAMES.CG}-v{version}")
            return f"fluidos/{MODEL_HF_NAMES.CG}-v{version}"
        if self.weights_path:
            torch.save({'model_state_dict': self.orchestrator.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                        },
                       self.weights_path.joinpath(MODEL_CHECKPOINT_NAMES.CG_LAST))

            return self.weights_path
        return None

    def __save_intetmediate_results(self,
                                    train_loss: float,
                                    valid_loss: float,
                                    epoch: int | str,
                                    ) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir.joinpath(f"{MODEL_CHECKPOINT_NAMES.CG_INTERMEDIATE}_{epoch}.pt")  # TODO fix to canpture general idea of this variable
        torch.save({'epoch': epoch,
                    'model_state_dict': self.orchestrator.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': valid_loss},
                   self.checkpoint_path)


class CGDataset(Dataset):  # type: ignore

    padding_value = -1

    def __init__(self, dataframe: pd.DataFrame, device: str = "cpu", emb_batch_size: int = 10) -> None:
        self.embedding_model: SentenceTransformer = SentenceTransformer("distiluse-base-multilingual-cased-v2",
                                                                        device=device)
        self.device = device
        self.df = dataframe
        embeddings = self.__from_text_to_embedding(self.df[FLUIDOS_COL_NAMES.POD_MANIFEST].tolist(),
                                                   batch_size=emb_batch_size)
        self.pod_embeddings = embeddings.to(device).squeeze(0)

        self.pod_cpu = torch.tensor(self.df[FLUIDOS_COL_NAMES.POD_CPU].tolist(), device=self.device).unsqueeze(1)
        self.pod_memory = torch.tensor(self.df[FLUIDOS_COL_NAMES.POD_MEMORY].tolist(), device=self.device).unsqueeze(1)
        # TODO: add translation of machine_id to config id

        non_relevant_configs_list: list[torch.Tensor] = [torch.tensor(ast.literal_eval(item), device=self.device, dtype=torch.int32)
                                                         for item in self.df[FLUIDOS_COL_NAMES.NON_ACCEPTABLE_CANDIDATES]]
        self.non_relevant_configs: torch.Tensor = pad_sequence(non_relevant_configs_list, batch_first=True).to(self.device)

        relevant_configs_list: list[torch.Tensor] = [torch.tensor(ast.literal_eval(item), device=self.device, dtype=torch.int32)
                                                     for item in self.df[FLUIDOS_COL_NAMES.ACCEPTABLE_CANDIDATES]]
        self.relevant_configs: torch.Tensor = pad_sequence(relevant_configs_list, batch_first=True).to(self.device)

        #TODO this will only work for feedback augmentation because of the dependency on TARGET_MOST_OPTIMAL_TEMPLATE_ID
        self.target: torch.Tensor = torch.stack(
            [torch.tensor(item, device=self.device, dtype=torch.float32) for item in self.df[FLUIDOS_COL_NAMES.TARGET_MOST_OPTIMAL_TEMPLATE_ID]],
            dim=0
        ).to(self.device)

    def __from_text_to_embedding(self, pods_list: list[str], batch_size: int = 10) -> torch.Tensor:
        embeddings_list: list[torch.Tensor] = []
        pbar = tqdm.tqdm(range(0, len(pods_list), batch_size), desc="Embeddings generation")
        for i in pbar:
            pod_batch = pods_list[i:i + batch_size]
            embedding_batch = self.embedding_model.encode(pod_batch)
            for j in range(embedding_batch.shape[0]):
                embeddings_list.append(torch.tensor(embedding_batch[j, :]).unsqueeze(0))
        embeddings_tensor = torch.stack(embeddings_list, dim=1)
        return embeddings_tensor

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Extracts dataset item by index

        Args:
            index (int): index of item

        Returns:
            Union[feature, target]:
                feature (Union[input_vector, relevant_configs, non_relevant_configs]):
                        input_vector (torch.Tensor): tensor of pod embedding vector amd pod location encoding
                        relevant_configs (torch.tensor): tensor of relevant config ids
                        non_relevant_configs (torch.tensor): tensor of non relevant config ids
                target (torch.Tensor):  tensor of size of condifurations dictionary
                  with maximum value index refers to the predicted configuration id

        """
        return (self.pod_embeddings[index], self.relevant_configs[index], self.non_relevant_configs[index]), self.target[index]

    def __len__(self) -> int:
        return self.pod_embeddings.shape[0]
