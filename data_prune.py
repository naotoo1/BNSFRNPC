from dataclasses import dataclass
from typing import Union
import numpy as np
import torch
from bns import (
    Prune,
    PruneMode,
    PruneType,
)
from tqdm import tqdm
from keras.applications import ResNet152V2
from keras.applications.resnet_v2 import preprocess_input


@dataclass
class ExtractedFeatures:
    features: Union[np.ndarray, list]
    image_names: list[str]


def get_extracted_image_features(
    dataset: torch.Tensor,
) -> ExtractedFeatures:
    extracted_features, image_name = [], []
    feature_extractor = ResNet152V2(
        weights="imagenet",
        include_top=False,
    )
    for image_index in tqdm(dataset):
        img = np.expand_dims(image_index, axis=0)
        img = preprocess_input(img)
        features = feature_extractor.predict(img)
        features = features.flatten()
        extracted_features.append(features)
        image_name.append(image_index)
    return ExtractedFeatures(
        features=extracted_features,
        image_names=image_name,
    )


def get_prune_data(
    data_name: str,
    dataset,
    labels,
    ssl_type: str,
    prune_fraction: float,
    prune_mode: str = PruneMode.EASY,  # pegged
    prune_type: str = PruneType.BOTH,  # pegged
    random_state: int = 40,
    feature_extraction: bool = False,
):
    match (data_name, feature_extraction):
        case ("cifar10", True):
            data_features = get_extracted_image_features(
                dataset=dataset,
            ).features
        case _:
            data_features = dataset

    prune: Prune = Prune(
        prune_type=True,
        ssl_type=ssl_type,
        number_cluster=len(np.unique(labels)),
        random_state=random_state,
        dataset=data_features,
    )

    return prune.get_prune(
        prune_fraction=prune_fraction,
        pruned_mode=prune_mode,
        prune_type=prune_type,
    )
