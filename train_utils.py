from dataclasses import dataclass
from enum import Enum
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from data import DATA, Sampling
from bns import SSLType, PruneMode

from data_prune import get_prune_data


class Dataset(str, Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    BREASTCANCER = "breast_cancer"
    COD_RNA = "cod-rna"


@dataclass(slots=True)
class TensorSet:
    X_train: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor


@dataclass(slots=True)
class EasyPrunedSpace:
    input_features: torch.Tensor
    input_labels: torch.Tensor


@dataclass(slots=True)
class HardPrunedSpace:
    input_features: torch.Tensor
    input_labels: torch.Tensor


@dataclass(slots=True)
class PrunedSpace:
    easy: EasyPrunedSpace
    hard: HardPrunedSpace


def prepare_data(
    dataset: str,
    test_size: float = 0.2,
) -> TensorSet:
    sample_size = 400
    match dataset:
        case Dataset.MNIST:
            input_features, labels = DATA().mnist
            input_features, labels = input_features[:4000], labels[:4000]
        case Dataset.CIFAR10:
            input_features, labels = DATA(  # type: ignore
                # sample=Sampling.RANDOM.value, sample_size=sample_size
            ).cifar_10
        case _:
            raise NotImplementedError(
                "dataset: choose either mnist or cifar10",
            )

    data_size = len(labels)
    test_split = int(np.floor(test_size * data_size))
    train_split = int(data_size - test_split)

    train_set, train_labels = input_features[:train_split], labels[:train_split]
    test_set, test_labels = input_features[-test_split:], labels[-test_split:]
    train_set = torch.div(train_set, 255)
    test_set = torch.div(test_set, 255)

    match dataset:
        case Dataset.CIFAR10:
            shuffle_list = list(np.arange(sample_size * 10))
            random.shuffle(shuffle_list)
            data_size = len(shuffle_list)
            test_split = int(np.floor(test_size * data_size))
            train_split = int(data_size - test_split)

            train_s, test_s = shuffle_list[:train_split], shuffle_list[-test_split:]

            train_set, train_labels = input_features[train_s], labels[train_s]
            test_set, test_labels = input_features[test_s], labels[test_s]
            train_set = torch.div(train_set, 255)
            test_set = torch.div(test_set, 255)

    return TensorSet(
        X_train=train_set.float(),
        X_test=test_set.float(),
        y_train=train_labels,
        y_test=test_labels,
    )


def get_tensor_data(
    dataset: str,
    test_size: float = 0.3,
    random_state: int = 4,
) -> TensorSet:
    match dataset:
        case Dataset.BREASTCANCER:
            input_features, labels = DATA().breast_cancer
            X_train, X_test, y_train, y_test = train_test_split(
                np.array(input_features),
                np.array(labels),
                test_size=test_size,
                random_state=random_state,
            )
            train_scaler, test_scaler = MinMaxScaler(), MinMaxScaler()
            X_train = train_scaler.fit_transform(X_train)
            X_test = test_scaler.fit_transform(X_test)
        case Dataset.COD_RNA:
            X_train, y_train, X_test, y_test = DATA().cod_rna
            pass
        case _:
            raise NotImplementedError(
                "get_tensor_data: choose breast_cancer",
            )

    x_input = torch.from_numpy(X_train).to(torch.float32)
    y_label = torch.from_numpy(y_train).to(torch.float32)
    x_input_test = torch.from_numpy(X_test).to(torch.float32)
    y_label_test = torch.from_numpy(y_test).to(torch.long)

    return TensorSet(
        X_train=x_input,
        X_test=x_input_test,
        y_train=y_label,
        y_test=y_label_test,
    )


def get_data(
    dataset: str,
    test_size: float = 0.3,
    random_state: int = 4,
) -> TensorSet:
    tabular_set = [Dataset.BREASTCANCER.value, Dataset.COD_RNA.value]
    condition1 = dataset in tabular_set
    match condition1:
        case True:
            return get_tensor_data(
                dataset=dataset,
                test_size=test_size,
                random_state=random_state,
            )

        case False:
            return prepare_data(
                dataset=dataset,
                test_size=test_size,
            )


def get_prunning(
    dataset: str,
    input_features: torch.Tensor,
    input_labels: torch.Tensor,
    ssl_type: str = SSLType.HCM,
    prune_fraction: float = 0.2,
    prune_mode: str = PruneMode.EASY,
    feature_extraction: bool = True,
) -> EasyPrunedSpace | HardPrunedSpace | PrunedSpace:
    with torch.no_grad():
        # input_features = input_features.detach().cpu().numpy()
        prune_indices = get_prune_data(
            data_name=dataset,
            dataset=input_features,
            labels=input_labels,
            ssl_type=ssl_type,
            prune_fraction=prune_fraction,
            feature_extraction=feature_extraction,
        )
    match prune_mode:
        case PruneMode.EASY:
            return EasyPrunedSpace(
                input_features=torch.Tensor(input_features[prune_indices[0]]),
                input_labels=torch.Tensor(input_labels[prune_indices[0]]),
            )
        case PruneMode.HARD:
            return HardPrunedSpace(
                input_features=torch.Tensor(input_features[prune_indices[1]]),
                input_labels=torch.Tensor(input_labels[prune_indices[1]]),
            )
        case PruneMode.BOTH:
            return PrunedSpace(
                easy=EasyPrunedSpace(
                    input_features=torch.Tensor(input_features[prune_indices[0]]),
                    input_labels=torch.Tensor(input_labels[prune_indices[0]]),
                ),
                hard=HardPrunedSpace(
                    input_features=torch.Tensor(input_features[prune_indices[1]]),
                    input_labels=torch.Tensor(input_labels[prune_indices[1]]),
                ),
            )

        case _:
            raise NotImplementedError("PruneMode: select either hard,easy or both")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
