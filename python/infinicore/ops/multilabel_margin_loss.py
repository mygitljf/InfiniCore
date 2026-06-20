import infinicore
from infinicore.tensor import Tensor


def multilabel_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
) -> Tensor:
    r"""Computes the multilabel margin loss."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.multilabel_margin_loss(
        input, target,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )
