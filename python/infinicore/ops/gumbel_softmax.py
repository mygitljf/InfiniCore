import infinicore
from infinicore.tensor import Tensor


def gumbel_softmax(logits: Tensor, tau=1.0, hard=False, eps=1e-10, dim=-1) -> Tensor:
    r"""Samples from the Gumbel-Softmax distribution."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
