import infinicore
from infinicore.tensor import Tensor


def combinations(input: Tensor, r: int = 2, with_replacement: bool = False) -> Tensor:
    r"""Compute combinations of length r of the elements in a 1D input tensor."""

    if (
        infinicore.use_ntops
        and input.device.type in ("cuda", "musa")
        and hasattr(infinicore.ntops.torch, "combinations")
    ):
        return infinicore.ntops.torch.combinations(
            input, r=r, with_replacement=with_replacement
        )

    raise NotImplementedError(
        "combinations requires the ntops backend on cuda/musa"
    )
