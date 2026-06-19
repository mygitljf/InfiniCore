import infinicore
from infinicore.tensor import Tensor


def slogdet(input: Tensor):
    r"""Computes the sign and log absolute value of the determinant."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.slogdet(input)
