"""Test Suite for Perceptual metric"""

from distance import lpips_distance
from train_utils import get_data, Dataset


access_data = get_data(
    dataset=Dataset.CIFAR10,
    test_size=0.3,
)
x_test = access_data.X_test
if torch.cuda.is_available():
    x_test = x_test.to("cuda")
else:
    x_test = x_test.to(torch.device("cpu"))
# x_test = x_test.to("cuda")


input_0 = x_test[0].reshape(1, 32, 32, 3)
input_1 = x_test[1].reshape(1, 32, 32, 3)
input_2 = x_test[2].reshape(1, 32, 32, 3)


def test_lpips_distance():
    """
    Test lpips distance as a semi-metric
    """

    perceptual_metric = lpips_distance(input_0, input_1, p="l2")
    perceptual_metric_symmetry = lpips_distance(input_1, input_0, p="l2")
    perceptual_metric_t_1 = lpips_distance(input_0, input_2, p="l2")
    perceptual_metric_t_2 = lpips_distance(input_1, input_2, p="l2")

    assert perceptual_metric > 0
    assert perceptual_metric == perceptual_metric_symmetry
    assert perceptual_metric_t_1 <= perceptual_metric + perceptual_metric_t_2
