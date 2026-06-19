import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    is_broadcast,
)

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

_TEST_CASES_DATA = [
    # (shape, strides, kernel_size, dilation, padding, stride)
    ((2, 3, 8, 8), None, (3, 3), 1, 0, 1),
    ((1, 4, 10, 12), None, (5, 3), 1, 1, (2, 1)),
    ((2, 2, 16, 16), None, (4, 4), 1, 0, (4, 4)),
    ((3, 6, 7, 9), None, (3, 2), 1, 0, (1, 1)),
    ((1, 8, 9, 11), None, (2, 3), 1, 1, (1, 2)),
    # dilation
    ((2, 5, 12, 6), None, (3, 3), 2, (2, 1), (2, 1)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float64: {"atol": 1e-8, "rtol": 1e-8},
}

_TENSOR_DTYPES = [
    infinicore.float16,
    infinicore.float32,
    infinicore.bfloat16,
    infinicore.float64,
]


def _ref_im2col(input, kernel_size, dilation=1, padding=0, stride=1):
    """Reference im2col using torch.nn.functional.unfold."""
    return torch.nn.functional.unfold(
        input,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )


def parse_test_cases():
    test_cases = []

    for shape, strides, kernel_size, dilation, padding, stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-3, "rtol": 1e-3})

            a_spec = TensorSpec.from_tensor(shape, strides, dtype, name="input")

            test_cases.append(
                TestCase(
                    inputs=[a_spec],
                    kwargs={
                        "kernel_size": kernel_size,
                        "dilation": dilation,
                        "padding": padding,
                        "stride": stride,
                    },
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"im2col OUT_OF_PLACE shape={shape} ks={kernel_size}",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("im2col")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return _ref_im2col(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.im2col(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
