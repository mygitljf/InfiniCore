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
    # (shape, reduction, strides_a, strides_b)
    ((7,), "none", None, None),
    ((3, 5), "none", None, None),
    ((2, 3, 4), "none", None, None),
    ((8, 1), "none", None, None),
    # strided (non-contiguous)
    ((5, 7), "none", (7, 1), (7, 1)),
    # broadcast
    ((4, 1), "none", None, None),
    # reduction="mean": produces a scalar
    ((128, 64), "mean", None, None),
    ((128, 64), "sum", None, None),
    # 3D shapes
    ((4, 8, 16), "none", None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.float64: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float32, infinicore.float16, infinicore.float64]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape = data[0]
        reduction = data[1]
        strides_a = data[2] if len(data) > 2 else None
        strides_b = data[3] if len(data) > 3 else None

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            a_spec = TensorSpec.from_tensor(shape, strides_a, dtype, name="a")
            b_spec = TensorSpec.from_tensor(shape, strides_b, dtype, name="b")

            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={"reduction": reduction},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"mse_loss reduction={reduction} - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("MSELoss")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.mse_loss(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.mse_loss(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
