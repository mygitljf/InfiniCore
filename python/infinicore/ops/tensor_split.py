import infinicore
from infinicore.tensor import Tensor


def tensor_split(input: Tensor, indices_or_sections, dim=0):
    r"""Split a tensor into multiple sub-tensors along a given dimension."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.tensor_split(input, indices_or_sections, dim=dim)
