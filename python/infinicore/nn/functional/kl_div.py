from typing import Optional

import infinicore
from infinicore.tensor import Tensor


def kl_div(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    r"""Compute Kullback-Leibler divergence loss."""

    if (
        infinicore.use_ntops
        and input.device.type in ("cuda", "musa")
        and hasattr(infinicore.ntops.torch, "kl_div")
    ):
        return infinicore.ntops.torch.kl_div(
            input,
            target,
            reduction=reduction,
            log_target=log_target,
        )

    raise NotImplementedError("kl_div requires the ntops backend on cuda/musa")
