import infinicore
from infinicore.tensor import Tensor


def unflatten(input: Tensor, dim: int, sizes) -> Tensor:
    r"""Expand a dimension of the input tensor over multiple dimensions."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.unflatten(input, dim, sizes)
