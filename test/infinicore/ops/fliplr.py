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
    # (shape, strides)
    ((3, 5), None),
    ((8, 16), None),
    ((2, 3, 4), None),
    ((4, 4, 4, 4), None),
    ((16, 64), None),
    # contiguous strides (safe on Iluvatar)
    ((3, 5), (5, 1)),
    ((2, 3, 4), (12, 4, 1)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 0},
    infinicore.float32: {"atol": 0, "rtol": 0},
    infinicore.bfloat16: {"atol": 0, "rtol": 0},
    infinicore.int32: {"atol": 0, "rtol": 0},
    infinicore.int64: {"atol": 0, "rtol": 0},
}

_TENSOR_DTYPES = [infinicore.float32, infinicore.float16, infinicore.bfloat16, infinicore.int32, infinicore.int64]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape = data[0]
        in_strides = data[1] if len(data) > 1 else None

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})
            in_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="fliplr - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("FlipLR")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.fliplr(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.fliplr(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
