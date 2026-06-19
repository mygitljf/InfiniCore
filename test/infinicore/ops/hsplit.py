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


# (shape, dtype, indices_or_sections, output_count)
_TEST_CASES_DATA = [
    ((4, 8), infinicore.float32, 2, 2),
    ((4, 9), infinicore.float32, 3, 3),
    ((6,), infinicore.float32, 3, 3),
    ((4, 10), infinicore.float32, [3, 7], 3),
    ((2, 6, 6), infinicore.float32, 3, 3),
    ((4, 8), infinicore.float16, 2, 2),
    ((4, 8), infinicore.int32, 2, 2),
    ((4, 8), infinicore.int64, 4, 4),
    ((8,), infinicore.float32, 4, 4),
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
    for shape, dtype, ios, out_count in _TEST_CASES_DATA:
        tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})
        input_spec = TensorSpec.from_tensor(shape, None, dtype)

        test_cases.append(
            TestCase(
                inputs=[input_spec],
                kwargs={"indices_or_sections": ios},
                output_spec=None,
                comparison_target=None,
                tolerance=tol,
                description=f"hsplit - sections={ios}",
                output_count=out_count,
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Hsplit")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, input, indices_or_sections):
        return torch.hsplit(input, indices_or_sections)

    def infinicore_operator(self, input, indices_or_sections):
        return infinicore.hsplit(input, indices_or_sections)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
