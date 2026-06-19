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
    # (shape, strides, dim, sizes)
    # 2D
    ((2, 6), None, 1, (2, 3)),
    ((3, 12), None, 1, (3, 4)),
    # 3D
    ((2, 3, 8), None, 2, (2, 4)),
    ((2, 6, 4), None, 1, (2, 3)),
    # dim=0
    ((8, 3), None, 0, (2, 4)),
    # negative dim
    ((2, 3, 8), None, -1, (2, 4)),
    # 4D
    ((2, 3, 8, 5), None, 2, (2, 4)),
    # single-element sizes
    ((2, 6, 4), None, 1, (6,)),
    # 1D input
    ((12,), None, 0, (3, 4)),
    # multiple sub-dims
    ((2, 24, 3), None, 1, (2, 3, 4)),
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

    for shape, strides, dim, sizes in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-3, "rtol": 1e-3})

            a_spec = TensorSpec.from_tensor(shape, strides, dtype, name="input")

            test_cases.append(
                TestCase(
                    inputs=[a_spec],
                    kwargs={"dim": dim, "sizes": sizes},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"unflatten OUT_OF_PLACE shape={shape} dim={dim} sizes={sizes}",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("unflatten")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        dim = kwargs.pop("dim")
        sizes = tuple(kwargs.pop("sizes"))
        return torch.unflatten(*args, dim, sizes, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        dim = kwargs.pop("dim")
        sizes = tuple(kwargs.pop("sizes"))
        return infinicore.unflatten(*args, dim, sizes, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
