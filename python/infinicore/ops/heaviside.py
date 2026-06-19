import infinicore
from infinicore.tensor import Tensor


def heaviside(input: Tensor, values: Tensor, *, out=None) -> Tensor:
    r"""Computes the Heaviside step function for each element in input."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.heaviside(input, values, out=out)
