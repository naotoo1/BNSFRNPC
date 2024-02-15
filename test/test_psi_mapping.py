"""test suite for psi mapping """

import unittest
from typing import Tuple
import torch
import numpy as np
from distance import _get_lpips_model
from train_utils import get_data


# get feature embeder
model = _get_lpips_model()

# get image data
access_data = get_data(dataset="cifar10", test_size=0.3)
if torch.cuda.is_available():
    x_test = access_data.X_test.to("cuda")
else:
    x_test = access_data.X_test.to(torch.device("cpu"))
# x_test = access_data.X_test.to("cuda")

# reshape the input
input_0 = x_test[0].reshape(1, 32, 32, 3)
x = input_0.permute(0, 3, 2, 1)

# extract the activations
features = model.features(x)


class TestPSI(unittest.TestCase):
    def setUp(self):
        self.features: Tuple[torch.Tensor, ...] = features
        self.epsilon: float = 1e-10

    def test_normalization_factor_1(self):
        for feature_layer in features:
            normalization_factor_1 = (
                torch.sqrt(torch.sum(feature_layer**2, dim=1, keepdim=True))
                + self.epsilon
            )
            if torch.cuda.is_available():
                self.assertIsInstance(
                normalization_factor_1[0][0][0][0], torch.cuda.FloatTensor
                )
            else:
                self.assertIsInstance(
                normalization_factor_1[0][0][0][0], torch.FloatTensor
                )
            # self.assertIsInstance(
            #     normalization_factor_1[0][0][0][0], torch.cuda.FloatTensor
            # )
            self.assertEqual(torch.sum(normalization_factor_1, dim=0).any(), 1)

    def test_normalization_factor_2(self):

        normalization_factor_2 = [
            np.sqrt(feature_layer.size()[2] * feature_layer.size()[3])
            for feature_layer in features
        ]

        normalization_factor_2 = torch.Tensor(normalization_factor_2)

        self.assertEqual(normalization_factor_2[0], 16.0)
        self.assertEqual(normalization_factor_2[1], 8.0)
        self.assertEqual(normalization_factor_2[2], 4.0)
        self.assertEqual(normalization_factor_2[3], 4.0)
        self.assertEqual(normalization_factor_2[4], 4.0)

    def test_norm_factor(self):
        for feature_layer in features:
            normalization_factor_1 = (
                torch.sqrt(torch.sum(feature_layer**2, dim=1, keepdim=True))
                + self.epsilon
            )

            normalization__factor_2 = np.sqrt(
                feature_layer.size()[2] * feature_layer.size()[3]
            )

            norm_factor = normalization_factor_1 * normalization__factor_2

            self.assertNotEqual(normalization_factor_1.any(), normalization__factor_2)
            self.assertEqual(norm_factor.shape[0], 1)
            self.assertEqual(len(norm_factor.size()), 4)
    
    def test_channel_wise_normalization(self):
        for feature_layer in features:
            normalization_factor_1 = (
                torch.sqrt(torch.sum(feature_layer**2, dim=1, keepdim=True))
                + self.epsilon
            )

            normalization__factor_2 = np.sqrt(
                feature_layer.size()[2] * feature_layer.size()[3]
            )

            norm_factor = normalization_factor_1 * normalization__factor_2

            normalised_channels = feature_layer / norm_factor

            flattened_normalised_channels = normalised_channels.view(
                feature_layer.size()[0], -1
            )

            self.assertIsInstance(flattened_normalised_channels, torch.Tensor)
            self.assertEqual(flattened_normalised_channels.shape[0], 1)

    def tearDown(self):
        del self.features, self.epsilon
