import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, TensorSpec, TestCase, GenericTestRunner

_TEST_CASES_DATA = [
    ((2, 3, 15, 15), None, (3, 3), (5, 5), False),
    ((1, 4, 16, 14), (896, 224, 14, 1), (4, 3), (4, 5), False),
    ((2, 2, 17, 19), None, (5, 5), (7, 6), False),
    ((3, 6, 9, 11), None, (2, 2), (4, 5), False),
    ((1, 8, 20, 20), (3200, 400, 20, 1), (3, 3), (6, 6), False),
    ((2, 5, 12, 10), None, (4, 3), (3, 3), False),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def _fmp2d_start(oh, ow, sh, sw, kH, kW, inH, inW, outH, outW):
    """Integer-arithmetic start positions, matching the kernel ILUVATAR_ALPHA branch."""
    numer_h = inH - kH
    denom_h = outH - 1 if outH > 1 else 1
    sample_h_scaled = int(sh * numer_h)
    start_h = (oh * numer_h + sample_h_scaled) // denom_h - sample_h_scaled // denom_h
    if oh == outH - 1:
        start_h = inH - kH

    numer_w = inW - kW
    denom_w = outW - 1 if outW > 1 else 1
    sample_w_scaled = int(sw * numer_w)
    start_w = (ow * numer_w + sample_w_scaled) // denom_w - sample_w_scaled // denom_w
    if ow == outW - 1:
        start_w = inW - kW

    return start_h, start_w


def _manual_fractional_max_pool2d(input, kernel_size, output_size, _random_samples, return_indices=False):
    """Ground truth using integer arithmetic (avoids fp32/double precision discrepancy
    between GPU fp32 kernel and CPU double-precision torch on Iluvatar CoreX)."""
    inH, inW = input.shape[-2], input.shape[-1]
    kH, kW = kernel_size
    outH, outW = output_size
    squeeze = input.ndim == 3
    inp = input.unsqueeze(0) if squeeze else input
    N, C = inp.shape[0], inp.shape[1]
    rs = _random_samples
    out = torch.empty(N, C, outH, outW, dtype=inp.dtype, device=inp.device)

    for n in range(N):
        for c in range(C):
            sh, sw = float(rs[n, c, 0]), float(rs[n, c, 1])
            for oh in range(outH):
                for ow in range(outW):
                    start_h, start_w = _fmp2d_start(oh, ow, sh, sw, kH, kW, inH, inW, outH, outW)
                    window = inp[n, c, start_h:start_h + kH, start_w:start_w + kW]
                    out[n, c, oh, ow] = window.max()

    if squeeze:
        out = out.reshape(out.shape[1:])
    return out


def parse_test_cases():
    cases = []
    _gen = torch.Generator(device="cpu")
    _gen.manual_seed(42)

    for in_shape, in_strides, kernel_size, out_size, return_indices in _TEST_CASES_DATA:
        n_batch = 1 if len(in_shape) == 3 else in_shape[0]
        n_channels = in_shape[-3]

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            dt_name = str(dtype).rsplit(".", 1)[-1]
            test_dtype = getattr(torch, dt_name, torch.float32)
            in_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)
            random_samples = torch.rand(
                (n_batch, n_channels, 2),
                generator=_gen,
                dtype=test_dtype,
                device="cpu",
            ).cuda()
            kwargs = {
                "kernel_size": kernel_size,
                "output_size": out_size,
                "return_indices": return_indices,
                "_random_samples": random_samples,
            }
            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="FractionalMaxPool2d - OUT_OF_PLACE",
                )
            )

    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("FractionalMaxPool2d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return _manual_fractional_max_pool2d(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.fractional_max_pool2d(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
