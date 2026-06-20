import infinicore
from infinicore.tensor import Tensor


def frac(input: Tensor, *, out=None) -> Tensor:
    r"""Computes the fractional portion of each element in input."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.frac(input, out=out)
