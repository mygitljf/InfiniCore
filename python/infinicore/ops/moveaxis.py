import infinicore
from infinicore.tensor import Tensor


def moveaxis(input: Tensor, source, destination) -> Tensor:
    r"""Move axis(es) of a tensor to new position(s)."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.moveaxis(input, source, destination)
