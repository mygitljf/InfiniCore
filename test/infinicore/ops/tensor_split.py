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
    # (shape, strides, indices_or_sections, dim)
    # 1D integer split
    ((6,), None, 2, 0),
    ((7,), None, 3, 0),  # uneven
    ((10,), None, 2, 0),
    # 1D strided
    ((10,), (2,), 2, 0),
    # 2D
    ((3, 8), None, 2, 0),
    ((3, 8), None, 2, 1),
    ((4, 6), None, 2, 1),
    # 2D index-based
    ((10,), None, [3, 6], 0),
    # 2D strided
    ((3, 8), (16, 2), 2, 1),
    # 3D
    ((2, 3, 4), None, 2, 1),
    ((2, 3, 4), None, 2, 2),
    # negative dim
    ((3, 8), None, 2, -1),
    # split into 1
    ((10,), None, 1, 0),
    # 4D
    ((2, 3, 4, 5), None, 2, 2),
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


def _output_count(indices_or_sections, dim, shape):
    """Compute expected number of output tensors for tensor_split."""
    if isinstance(indices_or_sections, int):
        return indices_or_sections
    return len(indices_or_sections) + 1


def parse_test_cases():
    test_cases = []

    for shape, strides, indices_or_sections, dim in _TEST_CASES_DATA:
        n_outputs = _output_count(indices_or_sections, dim, shape)

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-3, "rtol": 1e-3})

            a_spec = TensorSpec.from_tensor(shape, strides, dtype, name="input")

            test_cases.append(
                TestCase(
                    inputs=[a_spec],
                    kwargs={"indices_or_sections": indices_or_sections, "dim": dim},
                    output_spec=None,
                    output_specs=None,
                    output_count=n_outputs,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"tensor_split OUT_OF_PLACE shape={shape} sections={indices_or_sections} dim={dim}",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("tensor_split")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        indices_or_sections = kwargs.pop("indices_or_sections")
        dim = kwargs.pop("dim", 0)
        return torch.tensor_split(*args, indices_or_sections, dim=dim, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        indices_or_sections = kwargs.pop("indices_or_sections")
        dim = kwargs.pop("dim", 0)
        return infinicore.tensor_split(*args, indices_or_sections, dim=dim, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
