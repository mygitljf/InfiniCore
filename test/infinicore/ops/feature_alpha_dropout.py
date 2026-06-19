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

# feature_alpha_dropout with training=False or p=0.0 is deterministic (identity).
# training=True with p>0 is random and cannot be compared exactly.

_TEST_CASES_DATA = [
    # (shape, strides, training, p_val)
    ((2, 4, 8, 8), None, False, 0.5),
    ((2, 4, 8, 8), (256, 64, 8, 1), False, 0.5),
    ((4, 8, 16), None, False, 0.5),
    ((8, 16, 4, 4), None, False, 0.5),
    ((2, 4, 8, 8), None, True, 0.0),
    ((4, 8, 16), None, True, 0.0),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 0},
    infinicore.float32: {"atol": 0, "rtol": 0},
    infinicore.float64: {"atol": 0, "rtol": 0},
}

_TENSOR_DTYPES = [infinicore.float32, infinicore.float64]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape = data[0]
        in_strides = data[1] if len(data) > 1 else None
        training = data[2]
        p_val = data[3]

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})
            in_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            desc = f"training={training} p={p_val}"

            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={"p": p_val, "training": training},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"feature_alpha_dropout - OUT_OF_PLACE ({desc})",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("FeatureAlphaDropout")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.feature_alpha_dropout(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.feature_alpha_dropout(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
