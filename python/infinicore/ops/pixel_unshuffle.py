import infinicore
from infinicore.tensor import Tensor


def pixel_unshuffle(input: Tensor, downscale_factor: int) -> Tensor:
    r"""Reverses the PixelShuffle operation by rearranging elements in a tensor."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.pixel_unshuffle(input, downscale_factor)
