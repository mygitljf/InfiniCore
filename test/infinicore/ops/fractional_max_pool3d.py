import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, TensorSpec, TestCase, GenericTestRunner

_TEST_CASES_DATA = [
    ((2, 3, 9, 9, 9), None, (3, 3, 3), (4, 4, 4), False),
    ((1, 4, 8, 10, 12), None, (2, 3, 2), (4, 4, 6), False),
    ((2, 2, 7, 11, 5), (770, 110, 55, 5, 1), (3, 2, 3), (3, 4, 2), False),
    ((3, 6, 5, 6, 7), None, (2, 2, 2), (3, 3, 4), False),
    ((1, 8, 10, 10, 10), None, (4, 3, 2), (5, 4, 5), False),
    ((2, 5, 12, 8, 6), None, (3, 3, 2), (4, 3, 2), False),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    _gen = torch.Generator(device="cpu")
    _gen.manual_seed(42)

    for in_shape, in_strides, kernel_size, out_size, return_indices in _TEST_CASES_DATA:
        n_batch = 1 if len(in_shape) == 4 else in_shape[0]
        n_channels = in_shape[-4]

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            dt_name = str(dtype).rsplit(".", 1)[-1]
            test_dtype = getattr(torch, dt_name, torch.float32)
            in_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)
            random_samples = torch.rand(
                (n_batch, n_channels, 3),
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
                    description="FractionalMaxPool3d - OUT_OF_PLACE",
                )
            )

    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("FractionalMaxPool3d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.fractional_max_pool3d(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.fractional_max_pool3d(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
