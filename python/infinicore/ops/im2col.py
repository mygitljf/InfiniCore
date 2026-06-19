import infinicore
from infinicore.tensor import Tensor


def im2col(
    input: Tensor,
    kernel_size,
    dilation=1,
    padding=0,
    stride=1,
) -> Tensor:
    r"""Extract sliding local blocks from a batched input tensor (im2col)."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.im2col(
        input,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
