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
    # (shape, dims, strides, dtype_list)
    ((7,), [0], None, [infinicore.float32, infinicore.float16, infinicore.int32, infinicore.int64]),
    ((3, 5), [0], None, [infinicore.float32]),
    ((3, 5), [1], None, [infinicore.float32]),
    ((3, 5), [0, 1], None, [infinicore.float32]),
    ((2, 3, 4), [0, 2], None, [infinicore.float32]),
    ((2, 3, 4), [1], None, [infinicore.float32]),
    ((4, 4), [-1], None, [infinicore.float32]),
    # strided
    ((4, 4), [0], (4, 1), [infinicore.float32]),
    ((5, 7), [0], None, [infinicore.float32]),
    # 3D permuted strides
    ((4, 8, 16), [0, 1], (128, 16, 1), [infinicore.float32]),
    # large contiguous
    ((16, 64), [0, 1], None, [infinicore.float32]),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 0},
    infinicore.float32: {"atol": 0, "rtol": 0},
    infinicore.bfloat16: {"atol": 0, "rtol": 0},
    infinicore.int32: {"atol": 0, "rtol": 0},
    infinicore.int64: {"atol": 0, "rtol": 0},
}


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape = data[0]
        dims = data[1]
        in_strides = data[2] if len(data) > 2 else None
        dtype_list = data[3]

        for dtype in dtype_list:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})
            in_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={"dims": dims},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"flip dims={dims} - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Flip")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.flip(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.flip(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
