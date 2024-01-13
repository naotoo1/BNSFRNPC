from dataclasses import dataclass
from foolbox.models import pytorch
from foolbox.utils import accuracy
from foolbox.attacks import projected_gradient_descent
from foolbox.attacks import LinfPGD, carlini_wagner
from prototorch.core.distances import (
    squared_euclidean_distance,
    lpnorm_distance,
)
import torch


@dataclass
class LowerBoundRTM:
    LRTE: float
    LRTA: float
    CTE: float
    CTA: float


@dataclass
class UpperBoundRTM:
    URTE: float
    URTA: float


def get_lowerbound_certification(
    model,
    data: torch.Tensor,
    labels: torch.LongTensor,
    epsilon: float,
    p_norm: int | str,
    device: str = "cpu",
) -> LowerBoundRTM:
    # load model if any or live certification
    match isinstance(model, str):
        case True:
            model = torch.load(model)
            model.eval()
        case _:
            pass

    fmodel = pytorch.PyTorchModel(
        model=model,
        bounds=(0, 1),
        device=device,
    )
    clean_acc = accuracy(fmodel, data, labels)

    # setup attacks
    match p_norm:
        case "linf":
            attack = LinfPGD()
            raw_advs, clipped_advs, success = attack(
                fmodel, data, labels, epsilons=epsilon
            )
        case "l2":
            attack = carlini_wagner.L2CarliniWagnerAttack()
            raw_advs, clipped_advs, success = attack(
                fmodel, data, labels, epsilons=epsilon
            )
        case "l1":
            attack = projected_gradient_descent.L1ProjectedGradientDescentAttack()
            raw_advs, clipped_advs, success = attack(
                fmodel, data, labels, epsilons=epsilon
            )
        case _:
            raise NotImplementedError(
                "evaluate with either l1, l2 or linf norm",
            )

    robust_test_error = success.float().mean(axis=-1).item()

    robust_test_accuracy = 1 - robust_test_error

    return LowerBoundRTM(
        LRTE=round(robust_test_error, 4),
        LRTA=round(robust_test_accuracy, 4),
        CTA=round(clean_acc, 4),
        CTE=round(1 - clean_acc, 4),
    )


def get_upperbound_certification(
    x_test: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.LongTensor,
    ppc: int,
    epsilon: float,
    p_norm: str | int | None = None,
    q_norm: str | int | None = None,
) -> float:
    match p_norm:
        case "l2":
            distance_space = lpnorm_distance(
                x=x_test,
                y=prototypes,
                p=2,
            )
            urte = get_urte(
                distance_space=distance_space,
                labels=labels,
                ppc=ppc,
                epsilon=epsilon,
                p_norm=p_norm,
                q_norm=q_norm,
                x_test=x_test,
            )
            score = len(urte) / len(x_test)
            return round(score, 4)

        case "linf":
            distance_space = lpnorm_distance(
                x=x_test,
                y=prototypes,
                p=float("inf"),
            )

            urte = get_urte(
                distance_space=distance_space,
                labels=labels,
                ppc=ppc,
                epsilon=epsilon,
                p_norm=p_norm,
                q_norm=q_norm,
                x_test=x_test,
            )
            score = len(urte) / len(x_test)
            return round(score, 4)

        case "l1":
            distance_space = lpnorm_distance(
                x=x_test,
                y=prototypes,
                p=1,
            )

            urte = get_urte(
                distance_space=distance_space,
                labels=labels,
                ppc=ppc,
                epsilon=epsilon,
                p_norm=p_norm,
                q_norm=q_norm,
                x_test=x_test,
            )
            score = len(urte) / len(x_test)
            return round(score, 4)

        case None:
            distance_space = squared_euclidean_distance(
                x=x_test,
                y=prototypes,
            )
            urte = get_urte(
                distance_space=distance_space,
                labels=labels,
                ppc=ppc,
                epsilon=epsilon,
                p_norm="l2",
                q_norm=q_norm,
                x_test=x_test,
            )
            score = len(urte) / len(x_test)
            return round(score, 4)
        case _:
            raise NotImplementedError(
                "compute the closest distance: use l2, linf or l1 as  pnorm"
            )


def get_urte(
    distance_space: torch.Tensor,
    labels: torch.LongTensor,
    ppc: int,
    epsilon: float,
    p_norm: str,
    q_norm: str,
    x_test: torch.Tensor,
) -> list[float]:
    margin = []
    p = get_certification_norms(p_norm)
    q = get_certification_norms(q_norm)
    cond, epsilon = p <= q, 2 * epsilon
    input_shape = torch.Tensor(list(x_test.shape[1:]))
    n = int(torch.prod(input_shape).item())

    for i, v in enumerate(distance_space):
        dist = distance_space[i].reshape(len(torch.unique(labels)), ppc)
        min_dist = torch.Tensor([torch.min(protos_dis) for protos_dis in dist])
        wp_wm_indices = torch.argsort(min_dist)[:2]

        if labels[i] == wp_wm_indices[0]:
            wp = wp_wm_indices[0]
            wm = wp_wm_indices[1]
        else:
            wp = wp_wm_indices[1]
            wm = wp_wm_indices[0]
        dp = min_dist[wp].item()
        dm = min_dist[wm].item()
        hypothesis_margin = dm - dp
        hypothesis_margin = round(hypothesis_margin, 4)

        match cond:
            case True:
                if hypothesis_margin <= epsilon:
                    margin.append(hypothesis_margin)
            case False:
                if (n ** (1 / p - 1 / q) * hypothesis_margin) <= epsilon:
                    margin.append(hypothesis_margin)
    return margin


def get_certification_norms(input_norm: str):
    match input_norm:
        case "l1":
            return 1
        case "l2":
            return 2
        case "linf":
            return float("inf")
        case _:
            raise NotImplementedError("use either l1,l2 or linf")


@dataclass(slots=True)
class LPN:
    lp_norm: str

    def get_lpnorms(self, x, y):
        match self.lp_norm:
            case "l2":
                return lpnorm_distance(x, y, 2)
            case "linf":
                return lpnorm_distance(x, y, float("inf"))
            case "l1":
                return lpnorm_distance(x, y, 1)
            case _:
                raise NotImplementedError(
                    "get_lpnorms:none of the cases did match",
                )
