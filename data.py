from enum import Enum
import random
from typing import Union
import numpy as np
import torch
from sklearn.datasets import (
    load_breast_cancer,
    make_moons,
    make_blobs,
)
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10


class Sampling(str, Enum):
    RANDOM = "random"
    FULL = "full"


def breast_cancer_dataset():
    data, labels = load_breast_cancer(return_X_y=True)
    return data, labels


def moons_dataset(random_state: Union[int, None] = None):
    data, labels = make_moons(
        n_samples=150,
        shuffle=True,
        noise=None,
        random_state=random_state,
    )
    return data, labels


def bloobs(random_state: Union[int, None] = None):
    data, labels = make_blobs(
        n_samples=[120, 80],
        centers=[[0.0, 0.0], [2.0, 2.0]],
        cluster_std=[1.2, 0.5],
        random_state=random_state,
        shuffle=False,
    )
    return data, labels


def mnist_dataset():
    train_dataset = MNIST(
        "~/datasets",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )
    test_dataset = MNIST(
        "~/datasets",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )
    return torch.cat([train_dataset.data, test_dataset.data]), torch.cat(
        [train_dataset.targets, test_dataset.targets]
    )


def cifar_10(sample: Sampling, size: int):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    return torch.cat(
        [torch.from_numpy(train_dataset.data), torch.from_numpy(test_dataset.data)]
    ), torch.cat(
        [
            torch.from_numpy(np.array(train_dataset.targets)),
            torch.from_numpy(np.array(test_dataset.targets)),
        ]
    )
    # classwise_labels = get_classwise_labels(full_train_labels)
    # Data,labels = get_random_inputs(
    #     full_train_ds,
    #     full_train_labels,
    #     classwise_labels,
    #     sample_size=size
    # )

    # if sample == 'full':
    #     return full_train_ds, full_train_labels
    # if sample == 'random':
    #     return Data,labels
    # raise RuntimeError("cifar-10:none of the cases match")


def cod_rna_datababse(dataset_path: str = "./T_data/"):
    train_data, test_data = dataset_path + "cod-rna.tr", dataset_path + "cod-rna.t"

    num_train, num_test, dim, num_train_orig, num_test_orig = 59535, 271617, 8, 0, 0
    X_train = np.zeros((num_train, dim))
    y_train = np.zeros(num_train)
    for i, line in enumerate(open(train_data, "r").readlines()):
        y_train[i] = int(float(line.split(" ")[0]))  # -1 or 1
        for s in line.split(" ")[1:]:
            coord_str, val_str = s.replace("\n", "").split(":")
            coord, val = int(coord_str) - 1, float(val_str)
            X_train[i, coord] = val
        num_train_orig += 1

    assert num_train == num_train_orig

    X_test = np.zeros((num_test, dim))
    y_test = np.zeros(num_test)

    for i, line in enumerate(open(test_data, "r").readlines()):
        y_test[i] = int(float(line.split(" ")[0]))
        for s in line.split(" ")[1:]:
            coord_str, val_str = s.replace("\n", "").split(":")
            coord, val = int(coord_str) - 1, float(val_str)
            X_test[i, coord] = val
        num_test_orig += 1

    assert num_test == num_test_orig

    X_train, X_test = min_max_scaler(X_train, X_test)

    n_test_final, n_train_final = 1000, 4000
    idx = np.random.permutation(num_test)[:n_test_final]
    idx_t = np.random.permutation(num_train)[:n_train_final]
    X_test, y_test = X_test[idx], y_test[idx]
    X_train, y_train = X_train[idx_t], y_train[idx_t]

    y_train = np.array([0 if i == -1 else 1 for i in y_train])
    y_test = np.array([0 if i == -1 else 1 for i in y_test])
    return X_train, y_train, X_test, y_test


def get_classwise_labels(
    full_labels: torch.Tensor,
    num_class: int = 10,
) -> np.ndarray:
    classwise_labels = []
    for class_label in range(num_class):
        for index, label in enumerate(full_labels):
            label = label.detach().cpu().numpy()
            if label == class_label:
                classwise_labels.append(index)
    return np.reshape(classwise_labels, (-1, 6000))


def get_random_inputs(
    full_train_ds: torch.Tensor,
    full_train_labels: torch.Tensor,
    classwise_labels: np.ndarray,
    sample_size: int = 1000,
):
    random_labels = []
    for class_ in classwise_labels:
        random.shuffle(class_)
        random_labels.append(class_[:sample_size])
    random_label_indices = np.array(random_labels)
    random_label_indices = random_label_indices.flatten()

    return torch.from_numpy(
        np.array([full_train_ds[index] for index in random_label_indices])
    ), torch.from_numpy(
        np.array([full_train_labels[index] for index in random_label_indices])
    )


def min_max_scaler(train_data, test_data):
    input_space_max = train_data.max(axis=0, keepdims=True)
    input_space_min = train_data.min(axis=0, keepdims=True)
    train_data = (train_data - input_space_min) / (input_space_max - input_space_min)
    test_data = (test_data - input_space_min) / (input_space_max - input_space_min)
    return train_data, test_data


class DATA:
    def __init__(
        self,
        sample: Sampling = Sampling.FULL,
        random: int = 4,
        sample_size: int = 1000,
    ):
        self.sample = sample
        self.random = random
        self.sample_size = sample_size

    @property
    def S_1(self):
        return moons_dataset(self.random)

    @property
    def S_2(self):
        return bloobs(self.random)

    @property
    def breast_cancer(self):
        return breast_cancer_dataset()

    @property
    def cod_rna(self):
        return cod_rna_datababse()

    @property
    def mnist(self):
        return mnist_dataset()

    @property
    def cifar_10(self):
        return cifar_10(self.sample, self.sample_size)
