import infinicore
from infinicore.tensor import Tensor


def corrcoef(input: Tensor) -> Tensor:
    r"""Estimate a Pearson correlation coefficient matrix."""

    if (
        infinicore.use_ntops
        and input.device.type in ("cuda", "musa")
        and hasattr(infinicore.ntops.torch, "corrcoef")
    ):
        return infinicore.ntops.torch.corrcoef(input)

    raise NotImplementedError(
        "corrcoef requires the ntops backend on cuda/musa"
    )
