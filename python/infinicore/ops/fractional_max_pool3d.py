import infinicore
from infinicore.tensor import Tensor


def fractional_max_pool3d(
    input: Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = False,
    _random_samples=None,
):
    r"""Applies 3D fractional max pooling over an input signal composed of several input planes."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.fractional_max_pool3d(
        input, kernel_size,
        output_size=output_size,
        output_ratio=output_ratio,
        return_indices=return_indices,
        _random_samples=_random_samples,
    )
