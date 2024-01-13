from enum import Enum
import torch
from torch.utils import data
import prototorch.core.initializers as pci
from torch.utils.data import DataLoader


class OmegaInitializers(str, Enum):
    PCALINEARTRANSFORMINITIALIZER = "PCALTI"
    ZEROSCOMPINITIALIZER = "ZCI"
    EYELINEARTRANSFORMINITIALIZER = "ETLI"
    ONESLINEARTRANSFORMINITIALIZER = "OLTI"
    RANDOMLINEARTRANSFORMINITIALIZER = "RLTI"
    LITERALLINEARTRANSFORMINITIALIZER = "LLTI"


class PrototypeInitializers(str, Enum):
    STRATIFIEDMEANSCOMPONENTINITIALIZER = "SMCI"
    STRATIFIEDSELECTIONCOMPONENTINITIALIZER = "SSCI"
    ZEROSCOMPINITIALIZER = "ZCI"
    MEANCOMPONENTINITIALIZER = "MCI"
    ONESCOMPONENTINITIALIZER = "OCI"
    RANDOMNORMALCOMPONENTINITIALIZER = "RNCI"
    LITERALCOMPONENTINITIALIZER = "LCI"
    CLASSAWARECOMPONENTINITIALIZER = "CACI"
    DATAAWARECOMPONENTINITIALIZER = "DACI"
    FILLVALUECOMPONENTINITIALIZER = "FVCI"
    SELECTIONCOMPONENTINITIALIZER = "SCI"
    STRATIFIEDSELECTIONCOMPONENTININTIALIZER = "SSCI"
    UNIFORMCOMPONENTINITIALIZER = "UCI"


def get_prototype_initializers(
    initializer: str,
    train_ds: data.TensorDataset,
    pre_initialised_prototypes: torch.Tensor | None = None,
    scale: float = 1,
    shift: float = 0,
    fill_value: float = 1,
    minimum: float = 0,
    maximum: float = 1,
    noise: float = 0,
):
    match initializer:
        case PrototypeInitializers.STRATIFIEDMEANSCOMPONENTINITIALIZER:
            return pci.SMCI(data=train_ds, noise=noise)
        case PrototypeInitializers.STRATIFIEDSELECTIONCOMPONENTINITIALIZER:
            return pci.SSCI(data=train_ds, noise=noise)
        case PrototypeInitializers.SELECTIONCOMPONENTINITIALIZER:
            return pci.SCI(data=train_ds.tensors[0], noise=noise)
        case PrototypeInitializers.MEANCOMPONENTINITIALIZER:
            return pci.MCI(data=train_ds.tensors[0], noise=noise)
        case PrototypeInitializers.ZEROSCOMPINITIALIZER:
            return pci.ZCI(shape=train_ds.tensors[0].size()[1])
        case PrototypeInitializers.ONESCOMPONENTINITIALIZER:
            return pci.OCI(shape=train_ds.tensors[0].size()[1])
        case PrototypeInitializers.LITERALCOMPONENTINITIALIZER:
            return pci.LCI(components=pre_initialised_prototypes)
        case PrototypeInitializers.CLASSAWARECOMPONENTINITIALIZER:
            return pci.CACI(data=train_ds, noise=noise)
        case PrototypeInitializers.DATAAWARECOMPONENTINITIALIZER:
            return pci.DACI(data=train_ds.tensors[0], noise=noise)
        case PrototypeInitializers.RANDOMNORMALCOMPONENTINITIALIZER:
            return pci.RNCI(
                shape=train_ds.tensors[0].size()[1], shift=shift, scale=scale
            )
        case PrototypeInitializers.FILLVALUECOMPONENTINITIALIZER:
            return pci.FVCI(train_ds.tensors[0].size()[1], fill_value)
        case PrototypeInitializers.UNIFORMCOMPONENTINITIALIZER:
            return pci.UCI(
                shape=train_ds.tensors[0].size()[1],
                minimum=minimum,
                maximum=maximum,
                scale=scale,
            )
        case _:
            raise RuntimeError(
                "get_prototype_initializers:none of the above matches",
            )


def get_omega_initializers(
    initializer: str | None,
    train_ds: data.TensorDataset,
    out_dim_first: bool = False,
    noise: float = 0,
    train_loader: DataLoader | None = None,
):
    match initializer:
        case OmegaInitializers.EYELINEARTRANSFORMINITIALIZER:
            return pci.ELTI(out_dim_first)
        case OmegaInitializers.PCALINEARTRANSFORMINITIALIZER:
            return pci.PCALTI(
                data=train_ds.tensors[0],
                noise=noise,
                out_dim_first=out_dim_first,
            )
        case OmegaInitializers.ZEROSCOMPINITIALIZER:
            return pci.ZLTI(out_dim_first)
        case OmegaInitializers.ONESLINEARTRANSFORMINITIALIZER:
            return pci.OLTI(out_dim_first)
        case OmegaInitializers.RANDOMLINEARTRANSFORMINITIALIZER:
            return pci.RLTI(out_dim_first)
