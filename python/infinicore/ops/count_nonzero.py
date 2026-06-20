import infinicore
from infinicore.tensor import Tensor


def count_nonzero(input: Tensor, dim=None) -> Tensor:
    r"""Count the number of non-zero values in the input tensor."""

    if (
        infinicore.use_ntops
        and input.device.type in ("cuda", "musa")
        and hasattr(infinicore.ntops.torch, "count_nonzero")
    ):
        return infinicore.ntops.torch.count_nonzero(input, dim=dim)

    raise NotImplementedError(
        "count_nonzero requires the ntops backend on cuda/musa"
    )
