import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorSpec,
    TestCase,
)

_TEST_CASES_DATA = [
    # (shape, downscale_factor, strides)
    ((1, 1, 8, 8), 2, None),
    ((2, 3, 8, 8), 2, None),
    ((1, 1, 9, 9), 3, None),
    ((4, 2, 16, 16), 4, None),
    ((2, 3, 12, 8), 2, None),
    # strided (permuted)
    ((2, 3, 8, 12), 2, (288, 96, 12, 1)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 0},
    infinicore.float32: {"atol": 0, "rtol": 0},
    infinicore.float64: {"atol": 0, "rtol": 0},
    infinicore.int32: {"atol": 0, "rtol": 0},
}

_TENSOR_DTYPES = [infinicore.float32, infinicore.float16, infinicore.float64, infinicore.int32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape = data[0]
        factor = data[1]
        in_strides = data[2] if len(data) > 2 else None

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})
            in_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={"downscale_factor": factor},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"pixel_unshuffle factor={factor} - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("PixelUnshuffle")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.pixel_unshuffle(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.pixel_unshuffle(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
