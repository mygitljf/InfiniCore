import infinicore
from infinicore.tensor import Tensor


def hsplit(input: Tensor, indices_or_sections):
    r"""Splits input horizontally into multiple tensors."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.hsplit(input, indices_or_sections)
