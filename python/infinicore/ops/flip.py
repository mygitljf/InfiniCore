import infinicore
from infinicore.tensor import Tensor


def flip(input: Tensor, dims) -> Tensor:
    r"""Reverses the order of elements in a tensor along the given dimensions."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.flip(input, dims)
