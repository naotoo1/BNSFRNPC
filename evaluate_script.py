import argparse
from enum import Enum
from dataclasses import dataclass
import numpy as np
import torch
from prototorch.core import lpnorm_distance
from evaluate import (
    get_lowerbound_certification,
    LowerBoundRTM,
    get_upperbound_certification,
)

from train_utils import (
    Dataset,
    get_data,
    seed_everything,
)


class Metric(str, Enum):
    URTE = "urte"
    LRTE = "lrte"
    LTRA = "lrta"
    CTA = "cta"
    CTE = "cte"


class LPNorms(str, Enum):
    L1 = "l1"
    L2 = "l2"
    LINF = "linf"


@dataclass(slots=True)
class TestSet:
    x_test: torch.Tensor
    y_test: torch.Tensor


@dataclass(slots=True)
class LPN:
    lp_norm: str

    def get_lpnorms(self, x, y):
        match self.lp_norm:
            case LPNorms.L2:
                return lpnorm_distance(x, y, 2)
            case LPNorms.LINF:
                return lpnorm_distance(x, y, np.inf)
            case LPNorms.L1:
                return lpnorm_distance(x, y, 1)
            case _:
                raise NotImplementedError(
                    "get_lpnorms:none of the cases did match",
                )


@dataclass(slots=True)
class EER:
    model: str
    dataset: str
    test_size: float
    epsilon: float
    p_norm: str
    q_norm: str
    metric: str | None
    random_state: int = 4

    @property
    def get_test_data(self) -> TestSet:
        access_data = get_data(
            dataset=self.dataset,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return TestSet(
            x_test=access_data.X_test,
            y_test=access_data.y_test,
        )

    @property
    def evaluate_empirical_robustness_lb(
        self,
    ) -> LowerBoundRTM:
        test_set = self.get_test_data
        metric = get_lowerbound_certification(
            model=self.model,
            data=test_set.x_test,
            labels=test_set.y_test,
            epsilon=self.epsilon,
            p_norm=self.p_norm,
        )
        return LowerBoundRTM(
            LRTE=metric.LRTE,
            LRTA=metric.LRTA,
            CTE=metric.CTE,
            CTA=metric.CTA,
        )

    @property
    def evaluate_empirical_robustness_ub(
        self,
    ) -> float:
        test_set = self.get_test_data
        model = torch.load(self.model)
        model.eval()
        ppc = int(len(model.prototypes) / len(torch.unique(test_set.y_test)))
        return get_upperbound_certification(
            x_test=test_set.x_test,
            prototypes=model.prototypes,
            labels=test_set.y_test,
            ppc=ppc,
            epsilon=self.epsilon,
            p_norm=self.p_norm,
            q_norm=self.q_norm,
        )


if __name__ == "__main__":
    seed_everything(seed=4)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument(
        "--dataset", type=str, required=False, default=Dataset.BREASTCANCER.value
    )
    parser.add_argument("--test_size", type=float, required=False, default=0.2)
    parser.add_argument("--p_norm", type=str, required=False, default=LPNorms.L2.value)
    parser.add_argument("--metric", type=str, required=False, default=None)
    parser.add_argument("--epsilon", type=float, required=False, default=0.025)
    parser.add_argument(
        "--train_norm", type=str, required=False, default=LPNorms.L2.value
    )

    model = parser.parse_args().model
    dataset = parser.parse_args().dataset
    p_norm = parser.parse_args().p_norm
    metric = parser.parse_args().metric
    test_size = parser.parse_args().test_size
    epsilon = parser.parse_args().epsilon
    train_norm = parser.parse_args().train_norm

    path = f"./weight_folder/{dataset}/{train_norm}_trained/{model}/{model}.pt"

    robust_evaluation = EER(
        model=path,
        dataset=dataset,
        test_size=test_size,
        epsilon=epsilon,
        p_norm=p_norm,
        metric=metric,
        q_norm=train_norm,
    )

    match metric:
        case Metric.LRTE:
            print(
                f"{Metric.LRTE.value} = {robust_evaluation.evaluate_empirical_robustness_lb.LRTE}",
            )
        case Metric.LTRA:
            print(
                f"{Metric.LTRA.value} = {robust_evaluation.evaluate_empirical_robustness_lb.LRTA}",
            )
        case Metric.CTE:
            print(
                f"{Metric.CTE.value} = {robust_evaluation.evaluate_empirical_robustness_lb.CTE}",
            )
        case Metric.CTA:
            print(
                f"{Metric.CTA.value} ={robust_evaluation.evaluate_empirical_robustness_lb.CTA}",
            )
        case Metric.URTE:
            print(
                f"{Metric.URTE.value} = {robust_evaluation.evaluate_empirical_robustness_ub}",
            )
        case None:
            print(
                f"{Metric.LRTE.value} = {robust_evaluation.evaluate_empirical_robustness_lb.LRTE}",
                f"{Metric.URTE.value} = {robust_evaluation.evaluate_empirical_robustness_ub}",
            )

        case _:
            raise NotImplementedError("metric:none of the cases match")
