import infinicore
from infinicore.tensor import Tensor


def channel_shuffle(input: Tensor, groups: int, *, out=None) -> Tensor:
    r"""Divide channels in a tensor into groups and rearrange them."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.channel_shuffle(input, groups, out=out)
