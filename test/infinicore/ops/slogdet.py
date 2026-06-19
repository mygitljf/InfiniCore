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


# (shape, dtype) — must be square matrices
_TEST_CASES_DATA = [
    ((3, 3), infinicore.float32),
    ((5, 5), infinicore.float32),
    ((2, 4, 4), infinicore.float32),
    ((6, 6), infinicore.float32),
    ((8, 8), infinicore.float32),
]

_TOLERANCE = {"atol": 1e-4, "rtol": 1e-4}


def parse_test_cases():
    test_cases = []
    for shape, dtype in _TEST_CASES_DATA:
        input_spec = TensorSpec.from_tensor(shape, None, dtype)

        test_cases.append(
            TestCase(
                inputs=[input_spec],
                kwargs={},
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE,
                description="slogdet - OUT_OF_PLACE",
                output_count=2,
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Slogdet")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, input):
        return torch.linalg.slogdet(input)

    def infinicore_operator(self, input):
        return infinicore.slogdet(input)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
