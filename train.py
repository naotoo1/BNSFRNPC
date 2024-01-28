import argparse
import os
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np
import prototorch.models as ps
import prototorch as pt
from prototorch.core import lpnorm_distance
from prototorch.models.extras import ltangent_distance
from distance import lpips_distance
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.warnings import PossibleUserWarning
from torch.utils import data
from torch.utils.data import DataLoader
from bns import SSLType, PruneMode
from evaluate import (
    get_lowerbound_certification,
    get_upperbound_certification,
)
from rnpc_initializers import (
    get_prototype_initializers,
    get_omega_initializers,
)
from train_utils import (
    Dataset,
    get_prunning,
    seed_everything,
    get_data,
)

torch.set_float32_matmul_precision(precision="high")
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class RNPC(str, Enum):
    GLVQ = "glvq"
    GTLVQ = "gtlvq"
    IGLVQ = "iglvq"
    IGTLVQ = "igtlvq"


class LPNorms(str, Enum):
    L1 = "l1"
    L2 = "l2"
    LINF = "linf"
    LPIPS_L1 = "lpips-l1"
    LPIPS_L2 = "lpips-l2"


class TransferFunctions(str, Enum):
    IDENTITY = "identity"
    SIGMOIDBETA = "sigmoid_beta"
    SWISHBETA = "swish_beta"


@dataclass(slots=True)
class TrainModelSummary:
    prototypes: list[torch.Tensor]
    omega_matrix: list[torch.Tensor]
    models: ps.GLVQ | ps.GTLVQ | ps.ImageGTLVQ | ps.ImageGLVQ


def train_rnpc(
    input_data: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    data_name: str,
    model_name: str,
    optimal_search: str,
    num_prototypes: int = 1,
    proto_lr: float = 0.01,
    bb_lr: float = 0.01,
    optimizer=torch.optim.Adam,
    proto_initializer: str = "SMCI",
    omega_matrix_initializer: str | None = "OLTI",
    save_model: bool = False,
    max_epochs: int = 2,
    noise: float = 0.1,
    batch_size: int = 128,
    num_workers: int = 4,
    lp_norm: str = LPNorms.L2,
) -> TrainModelSummary:
    prototypes, omega_matrix = [], []

    train_ds = data.TensorDataset(
        input_data,
        labels,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    num_classes = len(torch.unique(labels))
    condition = model_name == RNPC.GLVQ.value
    match (data_name, condition):
        case (Dataset.CIFAR10, False):
            input_dim = input_data.shape[-2]
            latent_dim = input_data.shape[-2]
            num_channels = input_data.shape[-1]
        case (Dataset.MNIST, False):
            input_dim = input_data.shape[-1]
            latent_dim = input_data.shape[-1]
            num_channels = None
        case (Dataset.BREASTCANCER, True):
            input_dim = None
            latent_dim = None
            num_channels = None
        case (Dataset.BREASTCANCER, False):
            input_dim = input_data.shape[1]
            latent_dim = input_data.shape[1] - 10
            num_channels = None
        case (Dataset.COD_RNA, False):
            input_dim = input_data.shape[1]
            latent_dim = input_data.shape[1] - 2
            num_channels = None
        case (Dataset.COD_RNA, True):
            input_dim = None
            latent_dim = None
            num_channels = None

        case _:
            raise NotImplementedError(
                "dataset: choose cifar10,minst, breast_cancer or cod-rna",
            )

    model = rnpc(
        model_name,
        train_ds,
        train_loader,
        input_dim,
        latent_dim,
        num_prototypes,
        num_classes,
        proto_lr,
        bb_lr,
        optimizer,
        proto_initializer,
        omega_matrix_initializer,
        noise,
        lp_norm=lp_norm,
        batch_size=batch_size,
        num_channels=num_channels,
    )

    trainer = model_trainer(
        search=optimal_search,
        max_epoch=max_epochs,
    )

    trainer.fit(model, train_loader)

    prototypes.append(model.prototypes)

    match model_name:
        case RNPC.IGTLVQ:
            omega_matrix.append(model.omega_matrix)
            saved_model_dir = f"./weight_folder/{data_name}/{model_name}"
        case _:
            saved_model_dir = (
                f"./weight_folder/{data_name}/{lp_norm}_trained/{model_name}"
            )

    if save_model:
        save_train_model(
            model_name=model_name,
            saved_model_dir=saved_model_dir,
            estimator=model,
        )

    return TrainModelSummary(
        prototypes=prototypes,
        models=model,
        omega_matrix=omega_matrix,
    )


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
            case "lpips-l2":
                return lpips_distance(x, y, "l2")
            case "lpips-l1":
                return lpips_distance(x, y, "l1")
            case "lpips-linf":
                return lpips_distance(x, y, "linf")
            case _:
                raise NotImplementedError(
                    "get_lpnorms:none of the cases did match",
                )


def rnpc(
    model: str,
    train_ds: data.TensorDataset,
    train_loader: DataLoader,
    input_dim: int | None,
    latent_dim: int | None,
    num_prototypes: int,
    num_classes: int,
    proto_lr: float,
    bb_lr: float,
    optimizer=torch.optim.Adam,
    proto_initializer: str = "SMCI",
    omega_matrix_initializer: str | None = "OLTI",
    noise: float = 0.1,
    lp_norm: str = "l2",
    batch_size: int = 256,
    num_channels: int | None = None,
) -> ps.GLVQ | ps.GTLVQ:
    prototype_initializer = get_prototype_initializers(
        initializer=proto_initializer,
        train_ds=train_ds,
        noise=noise,
    )

    omega_initializer = get_omega_initializers(
        initializer=omega_matrix_initializer,
        train_ds=train_ds,
        train_loader=train_loader,
    )

    get_lp_norms = LPN(lp_norm=lp_norm).get_lpnorms
    match model:
        case RNPC.GLVQ:
            hparams_glvq = dict(
                distribution={
                    "num_classes": num_classes,
                    "per_class": num_prototypes,
                },
                proto_lr=proto_lr,
                optimizer=optimizer,
                margin=0,
                transfer_fn="swish_beta",
                transfer_beta=10,
            )
            return ps.GLVQ(
                hparams_glvq,
                prototypes_initializer=prototype_initializer,
                distance_fn=get_lp_norms,
            )

        case RNPC.GTLVQ:
            hparams_gtlvq = dict(
                input_dim=input_dim,
                latent_dim=latent_dim,
                distribution={
                    "num_classes": num_classes,
                    "per_class": num_prototypes,
                },
                proto_lr=proto_lr,
                bb_lr=bb_lr,
                optimizer=optimizer,
                margin=0.0,
                transfer_fn="swish_beta",
                transfer_beta=10,
            )

            if omega_initializer:
                return ps.GTLVQ(
                    hparams_gtlvq,
                    prototypes_initializer=prototype_initializer,
                    omega_initializer=omega_initializer,
                    distance_fn=ltangent_distance,
                )
            return ps.GTLVQ(
                hparams_gtlvq,
                prototypes_initializer=prototype_initializer,
                distance_fn=ltangent_distance,
            )
        case RNPC.IGLVQ:
            hparams_glvq = dict(
                distribution={
                    "num_classes": num_classes,
                    "per_class": num_prototypes,
                },
                proto_lr=proto_lr,
                margin=0.0,
                transfer_fn="swish_beta",
                transfer_beta=10,
            )
            return ps.ImageGLVQ(
                hparams_glvq,
                optimizer=optimizer,
                prototypes_initializer=prototype_initializer,
                distance_fn=get_lp_norms,
            )

        case RNPC.IGTLVQ:
            cond = num_channels is None
            match cond:
                case True:
                    in_size = input_dim * input_dim
                case False:
                    in_size = input_dim * input_dim * num_channels

            hparams_igtlvq = dict(
                input_dim=in_size,
                latent_dim=latent_dim,
                distribution={
                    "num_classes": num_classes,
                    "per_class": num_prototypes,
                },
                proto_lr=proto_lr,
                bb_lr=bb_lr,
                margin=0.0,
                transfer_fn="swish_beta",
                transfer_beta=10,
            )
            return ps.ImageGTLVQ(
                hparams=hparams_igtlvq,
                optimizer=optimizer,
                prototypes_initializer=prototype_initializer,
                omega_initializer=pt.initializers.PCALinearTransformInitializer(
                    next(iter(train_loader))[0].reshape(batch_size, in_size)
                ),
                distance_fn=ltangent_distance,
            )

        case _:
            raise NotImplementedError(
                "specified_RNPC: none of the models did match",
            )


def save_train_model(
    *,
    saved_model_dir: str,
    model_name: str,
    estimator: ps.GLVQ | ps.GTLVQ,
):
    Path(saved_model_dir).mkdir(parents=True, exist_ok=True)
    try:
        torch.save(
            estimator,
            os.path.join(
                saved_model_dir,
                model_name + ".pt",
            ),
        )
    except AttributeError:
        pass


def model_trainer(
    search: str,
    max_epoch: int,
) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=max_epoch,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
        detect_anomaly=False,
        enable_model_summary=True,
        accelerator=search,
    )


if __name__ == "__main__":
    seed_everything(seed=4)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--data_name", type=str, required=False)
    parser.add_argument("--test_size", type=float, required=False, default=0.2)
    parser.add_argument(
        "--train_norm", type=str, required=False, default=LPNorms.LINF.value
    )
    parser.add_argument(
        "--test_norm", type=str, required=False, default=LPNorms.LINF.value
    )
    parser.add_argument("--proto_init", type=str, required=False, default="MCI")
    parser.add_argument("--omega_init", type=str, required=False, default="OLTI")
    parser.add_argument("--device", type=str, required=False, default="cpu")
    parser.add_argument(
        "--ssl_metric", type=str, required=False, default=SSLType.HCM.value
    )
    parser.add_argument("--batch_size", type=int, required=False, default=128)
    parser.add_argument("--test_epsilon", type=float, required=False, default=None)
    parser.add_argument("--num_proto", type=int, required=False, default=2)
    parser.add_argument("--prune_fraction", type=float, required=False, default=0.80)
    parser.add_argument(
        "--prune_mode", type=str, required=False, default=PruneMode.EASY.value
    )
    parser.add_argument("--feature_extraction", action="store_true", default=False)
    parser.add_argument("--prune", action="store_true", default=False)
    parser.add_argument("--max_epochs", type=int, required=False, default=10)
    parser.add_argument("--proto_lr", type=float, required=False, default=0.01)
    parser.add_argument("--omega_lr", type=float, required=False, default=0.01)
    parser.add_argument("--noise", type=float, required=False, default=0)

    model = parser.parse_args().model
    data_name = parser.parse_args().data_name
    train_norm = parser.parse_args().train_norm
    test_norm = parser.parse_args().test_norm
    test_size = parser.parse_args().test_size
    device = parser.parse_args().device
    batch_size = parser.parse_args().batch_size
    test_epsilon = parser.parse_args().test_epsilon
    proto_initializer = parser.parse_args().proto_init
    omega_matrix_initializer = parser.parse_args().omega_init
    num_proto = parser.parse_args().num_proto
    ssl_metric = parser.parse_args().ssl_metric
    prune_fraction = parser.parse_args().prune_fraction
    prune_mode = parser.parse_args().prune_mode
    feature_extraction = parser.parse_args().feature_extraction
    prune = parser.parse_args().prune
    max_epochs = parser.parse_args().max_epochs
    proto_lr = parser.parse_args().proto_lr
    omega_lr = parser.parse_args().omega_lr
    noise = parser.parse_args().noise

    match (data_name, test_epsilon):
        case (Dataset.MNIST, None):
            test_epsilon = 0.3
        case (Dataset.CIFAR10, None):
            test_epsilon = 8 / 255
        case (Dataset.BREASTCANCER, None):
            test_epsilon = 0.3
        case (Dataset.COD_RNA, None):
            test_epsilon = 0.025
        case _:
            raise NotImplementedError

    access_data = get_data(dataset=data_name, test_size=test_size)
    x_train, y_train = access_data.X_train, access_data.y_train
    x_test, y_test = access_data.X_test, access_data.y_test

    match prune:
        case True:
            pruned_results = get_prunning(
                dataset=data_name,
                input_features=x_train,
                input_labels=y_train,
                ssl_type=ssl_metric,
                prune_fraction=prune_fraction,
                prune_mode=prune_mode,
                feature_extraction=feature_extraction,
            )
            x_train, y_train = (
                pruned_results.input_features,
                pruned_results.input_labels,
            )
        case False:
            x_train, y_train = access_data.X_train, access_data.y_train

    tabular_data = [Dataset.BREASTCANCER.value, Dataset.COD_RNA.value]
    omega_matrix_initializer = (
        omega_matrix_initializer if data_name in tabular_data else "PCALTI"
    )
    learner = train_rnpc(
        input_data=x_train,
        labels=y_train,
        data_name=data_name,
        model_name=model,
        optimal_search=device,
        lp_norm=train_norm,
        batch_size=batch_size,
        proto_initializer=proto_initializer,
        omega_matrix_initializer=omega_matrix_initializer,
        save_model=True,
        num_prototypes=num_proto,
        noise=noise,
        proto_lr=proto_lr,
        bb_lr=omega_lr,
        max_epochs=max_epochs,
    )

    get_prototypes = learner.prototypes[0]
    trained_model = learner.models
    prototype_lables = trained_model.prototype_labels
    trained_model.eval()

    urte = get_upperbound_certification(
        x_test=x_test,
        prototypes=get_prototypes,
        labels=y_test,
        ppc=num_proto,
        epsilon=test_epsilon,
        p_norm=test_norm,
        q_norm=train_norm,
    )

    lrte = get_lowerbound_certification(
        model=trained_model,
        data=x_test,
        labels=y_test,
        epsilon=test_epsilon,
        p_norm=test_norm,
        q_norm=train_norm,
    )

    print("ROBUST EVALUATION SCORES")
    print(f"URTE = {urte}, LRTE = {lrte.LRTE} and CTE = {lrte.CTE}")
