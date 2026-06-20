import infinicore
from infinicore.tensor import Tensor


def scatter_add(input: Tensor, dim: int, index: Tensor, src: Tensor, *, out=None) -> Tensor:
    r"""Adds all values from src into input at the indices specified in index."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.scatter_add(input, dim, index, src, out=out)
