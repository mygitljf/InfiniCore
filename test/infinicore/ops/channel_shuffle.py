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
    # (shape, strides, groups)
    # 3D
    ((2, 4, 3), None, 2),
    ((2, 4, 3), (12, 3, 1), 2),
    # 4D
    ((2, 4, 3, 5), None, 2),
    ((2, 4, 3, 5), (60, 15, 5, 1), 2),
    ((2, 8, 4, 4), None, 4),
    ((2, 8, 4, 4), (128, 16, 4, 1), 4),
    # Identity groups (groups=1 or channels=groups)
    ((2, 4, 3, 5), None, 1),
    # Non-contiguous via strides
    ((2, 6, 4, 4), (192, 32, 8, 2), 3),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 2e-3, "rtol": 2e-3},
    infinicore.float32: {"atol": 2e-3, "rtol": 2e-3},
    infinicore.bfloat16: {"atol": 2e-3, "rtol": 2e-3},
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

    for shape, strides, groups in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-3, "rtol": 1e-3})

            if dtype in (infinicore.bool,):
                # bool channels: only test when channels divisible by groups
                if shape[1] % groups != 0:
                    continue

            a_spec = TensorSpec.from_tensor(shape, strides, dtype, name="input")
            a_supports_inplace = not is_broadcast(strides)

            # Out-of-place
            test_cases.append(
                TestCase(
                    inputs=[a_spec],
                    kwargs={"groups": groups},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"channel_shuffle OUT_OF_PLACE shape={shape} groups={groups}",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("channel_shuffle")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.channel_shuffle(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.channel_shuffle(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
