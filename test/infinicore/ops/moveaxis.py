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
    # (shape, strides, source, destination)
    # 2D
    ((3, 4), None, 0, 1),
    ((3, 4), None, 1, 0),
    ((3, 4), (8, 2), 0, 1),
    # 3D permutations
    ((2, 3, 4), None, 0, 1),
    ((2, 3, 4), None, 0, 2),
    ((2, 3, 4), None, 1, 0),
    ((2, 3, 4), None, 2, 0),
    ((2, 3, 4), None, 2, 1),
    # 3D strided
    ((2, 3, 4), (24, 8, 2), 0, 2),
    # 4D
    ((2, 3, 4, 5), None, 0, 3),
    ((2, 3, 4, 5), None, 3, 0),
    ((2, 3, 4, 5), None, 1, 2),
    # No-op
    ((3, 4), None, 0, 0),
    # Negative indices
    ((2, 3, 4), None, -1, 0),
    ((2, 3, 4), None, 0, -1),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float64: {"atol": 1e-8, "rtol": 1e-8},
    infinicore.int32: {"atol": 0, "rtol": 0},
    infinicore.int64: {"atol": 0, "rtol": 0},
    infinicore.bool: {"atol": 0, "rtol": 0},
}

_TENSOR_DTYPES = [
    infinicore.float16,
    infinicore.float32,
    infinicore.bfloat16,
    infinicore.float64,
    infinicore.int32,
    infinicore.int64,
    infinicore.bool,
]


def parse_test_cases():
    test_cases = []

    for shape, strides, source, destination in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-3, "rtol": 1e-3})

            a_spec = TensorSpec.from_tensor(shape, strides, dtype, name="input")

            test_cases.append(
                TestCase(
                    inputs=[a_spec],
                    kwargs={"source": source, "destination": destination},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"moveaxis OUT_OF_PLACE shape={shape} {source}->{destination}",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("moveaxis")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.moveaxis(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.moveaxis(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
