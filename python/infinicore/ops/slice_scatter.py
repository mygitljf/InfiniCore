import infinicore
from infinicore.tensor import Tensor


def slice_scatter(input: Tensor, src: Tensor, dim=0, start=None, end=None, step=1) -> Tensor:
    r"""Embeds src into input along a slice."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.slice_scatter(input, src, dim=dim, start=start, end=end, step=step)
