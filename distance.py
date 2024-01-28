from typing import Tuple, List
import numpy as np
import torch
from torch import nn
from torchvision import models
from prototorch.core.distances import (
    lpnorm_distance,
    squared_euclidean_distance,
)


class ImageNetNormalizer(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        mean = torch.tensor(self.mean, device=x.device)
        std = torch.tensor(self.std, device=x.device)

        return (x - mean[None, :, None, None]) / std[None, :, None, None]


class FeatureModel(nn.Module):
    """
    A classifier model which can produce layer features, output logits, or
    both.
    """

    normalizer: nn.Module
    model: nn.Module

    def __init__(self):
        super().__init__()
        self._allow_training = False
        self.eval()

    def features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Should return a tuple of features (layer1, layer2, ...).
        """

        raise NotImplementedError()

    def classifier(self, last_layer: torch.Tensor) -> torch.Tensor:
        """
        Given the final activation, returns the output logits.
        """

        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns logits for the given inputs.
        """

        return self.classifier(self.features(x)[-1])

    def features_logits(
        self,
        x: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Returns a tuple (features, logits) for the given inputs.
        """

        features = self.features(x)
        logits = self.classifier(features[-1])
        return features, logits

    def allow_train(self):
        self._allow_training = True

    def train(self, mode=True):
        if mode is True and not self._allow_training:
            raise RuntimeError("should not be in train mode")
        super().train(mode)


class AlexNetFeatureModel(FeatureModel):
    model: models.AlexNet

    def __init__(self, alexnet_model: models.AlexNet):
        super().__init__()
        self.normalizer = ImageNetNormalizer()
        self.model = alexnet_model.eval()

        assert len(self.model.features) == 13
        self.layer1 = nn.Sequential(self.model.features[:2])
        self.layer2 = nn.Sequential(self.model.features[2:5])
        self.layer3 = nn.Sequential(self.model.features[5:8])
        self.layer4 = nn.Sequential(self.model.features[8:10])
        self.layer5 = nn.Sequential(self.model.features[10:12])
        self.layer6 = self.model.features[12]

    def features(self, x):
        x = self.normalizer(x)

        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        x_layer5 = self.layer5(x_layer4)

        return (x_layer1, x_layer2, x_layer3, x_layer4, x_layer5)

    def classifier(self, last_layer):
        x = self.layer6(last_layer)
        if isinstance(self.model, CifarAlexNet):
            x = x
        else:
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
        return x


class CifarAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


def _get_lpips_model() -> FeatureModel:
    lpips_model: FeatureModel
    alexnet_model = CifarAlexNet()
    lpips_model = AlexNetFeatureModel(alexnet_model)
    if torch.cuda.is_available():
        lpips_model.cuda()
    else:
        lpips_model.to(torch.device("cpu"))
    state = torch.load("./alexnet_cifar.pt")
    lpips_model.load_state_dict(state["model"])
    lpips_model.eval()
    return lpips_model


def normalize_flatten_features(
    features: Tuple[torch.Tensor, ...],
    eps=1e-10,
) -> torch.Tensor:
    normalized_features: List[torch.Tensor] = []
    for feature_layer in features:
        norm_factor = (
            torch.sqrt(torch.sum(feature_layer**2, dim=1, keepdim=True)) + eps
        )
        normalized_features.append(
            (
                feature_layer
                / (
                    norm_factor
                    * np.sqrt(feature_layer.size()[2] * feature_layer.size()[3])
                )
            ).view(feature_layer.size()[0], -1)
        )
    return torch.cat(normalized_features, dim=1)


def lpips_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    p: str,
) -> torch.Tensor:
    model = _get_lpips_model()

    x = x.permute(0, 3, 2, 1)
    y = y.permute(0, 3, 2, 1)

    features_1 = model.features(x)
    features_2 = model.features(y)
    activations_x = normalize_flatten_features(features_1)
    activations_y = normalize_flatten_features(features_2)
    match p:
        case "l2":
            return lpnorm_distance(
                x=activations_x,
                y=activations_y,
                p=2,
            )
        case "l1":
            return lpnorm_distance(
                x=activations_x,
                y=activations_y,
                p=1,
            )
        case "linf":
            return lpnorm_distance(
                x=activations_x,
                y=activations_y,
                p=float("inf"),
            )
        case "sqr":
            return squared_euclidean_distance(
                x=activations_x,
                y=activations_y,
            )

        case _:
            raise NotImplemented("Use l2, l1, linf or sqr")
