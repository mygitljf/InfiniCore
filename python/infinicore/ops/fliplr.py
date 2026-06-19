import infinicore
from infinicore.tensor import Tensor


def fliplr(input: Tensor) -> Tensor:
    r"""Reverses the order of elements in a tensor along dim=1 (left-right)."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.fliplr(input)
