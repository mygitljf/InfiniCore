import infinicore
from infinicore.tensor import Tensor


def mse_loss(input: Tensor, target: Tensor, reduction="mean") -> Tensor:
    r"""Computes the mean squared error loss between input and target."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.mse_loss(input, target, reduction=reduction)
