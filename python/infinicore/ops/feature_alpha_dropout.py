import infinicore
from infinicore.tensor import Tensor


def feature_alpha_dropout(input: Tensor, p=0.5, training=True, inplace=False) -> Tensor:
    r"""Applies feature alpha dropout to the input tensor."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.feature_alpha_dropout(input, p=p, training=training, inplace=inplace)
