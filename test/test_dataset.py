""" BNSFRNPC dataset test suite"""

import unittest
import torch
import numpy as np
from train_utils import get_data, Dataset


class TestWDBC(unittest.TestCase):
    def setUp(self):

        self.data = get_data(
            dataset=Dataset.BREASTCANCER.value,
            test_size=0.3,
        )

    def test_size(self):

        self.assertEqual(
            len(self.data.X_test),
            171,
        )
        self.assertEqual(
            len(self.data.y_test),
            171,
        )
        self.assertEqual(
            len(self.data.X_train) + len(self.data.X_test),
            569,
        )
        self.assertEqual(
            len(self.data.y_train) + len(self.data.y_test),
            569,
        )

    def test_dimensions(self):
        self.assertEqual(
            self.data.X_train.shape[1],
            30,
        )
        self.assertEqual(
            self.data.X_test.shape[1],
            30,
        )

    def test_unique_labels(self):

        self.assertEqual(
            len(np.unique(self.data.y_train)),
            2,
        )
        self.assertEqual(
            len(np.unique(self.data.y_test)),
            2,
        )

    def tearDown(self):
        del self.data


class TestMNIST(unittest.TestCase):
    def setUp(self):
        self.data = get_data(
            dataset=Dataset.MNIST.value,
            test_size=0.3,
        )

    def test_size(self):

        self.assertEqual(
            self.data.X_test.shape[0],
            1200,
        )
        self.assertEqual(
            self.data.y_test.shape[0],
            1200,
        )
        self.assertEqual(
            len(self.data.X_train) + len(self.data.X_test),
            4000,
        )
        self.assertEqual(
            len(self.data.y_train) + len(self.data.y_test),
            4000,
        )

    def test_dimensions(self):

        self.assertEqual(
            self.data.X_train.shape[1],
            28,
        )
        self.assertEqual(
            self.data.X_train.shape[2],
            28,
        )
        self.assertEqual(
            self.data.X_test.shape[1],
            28,
        )
        self.assertEqual(
            self.data.X_test.shape[2],
            28,
        )

    def test_unique_labels(self):
        self.assertEqual(
            len(torch.unique(self.data.y_train)),
            10,
        )
        self.assertEqual(
            len(torch.unique(self.data.y_test)),
            10,
        )

    def tearDown(self):
        del self.data


class TestCIFAR10(unittest.TestCase):
    def setUp(self):

        self.data = get_data(
            dataset=Dataset.CIFAR10.value,
            test_size=0.3,
        )

    def test_size(self):

        self.assertEqual(
            self.data.X_test.shape[0],
            1200,
        )
        self.assertEqual(
            self.data.y_test.shape[0],
            1200,
        )
        self.assertEqual(
            len(self.data.X_train) + len(self.data.X_test),
            4000,
        )
        self.assertEqual(
            len(self.data.y_train) + len(self.data.y_test),
            4000,
        )

    def test_dimensions(self):

        self.assertEqual(
            self.data.X_train.shape[1],
            32,
        )
        self.assertEqual(
            self.data.X_train.shape[2],
            32,
        )
        self.assertEqual(
            self.data.X_test.shape[1],
            32,
        )
        self.assertEqual(
            self.data.X_test.shape[2],
            32,
        )
        self.assertEqual(
            self.data.X_train.shape[3],
            3,
        )
        self.assertEqual(
            self.data.X_test.shape[3],
            3,
        )

    def test_unique_labels(self):
        self.assertEqual(
            len(torch.unique(self.data.y_train)),
            10,
        )
        self.assertEqual(
            len(torch.unique(self.data.y_test)),
            10,
        )

    def tearDown(self):
        del self.data


class TestCODRNA(unittest.TestCase):
    def setUp(self):
        self.data = get_data(
            dataset=Dataset.COD_RNA.value,
            test_size=0.3,
        )

    def test_size(self):

        self.assertEqual(
            self.data.X_test.shape[0],
            1000,
        )
        self.assertEqual(
            self.data.y_test.shape[0],
            1000,
        )
        self.assertEqual(
            len(self.data.X_train) + len(self.data.X_test),
            5000,
        )
        self.assertEqual(
            len(self.data.y_train) + len(self.data.y_test),
            5000,
        )

    def test_dimensions(self):

        self.assertEqual(
            self.data.X_train.shape[1],
            8,
        )
        self.assertEqual(
            self.data.X_test.shape[1],
            8,
        )

    def test_unique_labels(self):
        self.assertEqual(
            len(torch.unique(self.data.y_train)),
            2,
        )
        self.assertEqual(
            len(torch.unique(self.data.y_test)),
            2,
        )

    def tearDown(self):
        del self.data
