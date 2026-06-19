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


# (input_shape, dim, start, end, step, input_strides, src_strides)
_TEST_CASES_DATA = [
    ((4, 4), 1, 1, 3, 1, None, None),
    ((4, 4), 0, 0, 2, 1, None, None),
    ((8, 6), 1, 0, 6, 2, None, None),
    ((2, 3, 4), 2, 1, 4, 1, None, None),
    ((10,), 0, 2, 8, 2, None, None),
    ((6, 6), 1, None, None, 1, None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 0},
    infinicore.float32: {"atol": 0, "rtol": 0},
    infinicore.bfloat16: {"atol": 0, "rtol": 0},
    infinicore.int32: {"atol": 0, "rtol": 0},
    infinicore.int64: {"atol": 0, "rtol": 0},
}

_DTYPES = [infinicore.float16, infinicore.float32, infinicore.int64]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, dim, start, end, step, in_strides, src_strides = data

        # Calculate src shape
        dim_size = shape[dim]
        lo = 0 if start is None else (start + dim_size if start < 0 else start)
        hi = dim_size if end is None else (end + dim_size if end < 0 else end)
        lo = max(0, min(lo, dim_size))
        hi = max(lo, min(hi, dim_size))
        src_len = len(range(lo, hi, step))
        src_shape = list(shape)
        src_shape[dim] = src_len
        src_shape = tuple(src_shape)

        for dtype in _DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})
            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype, name="input")
            src_spec = TensorSpec.from_tensor(src_shape, src_strides, dtype, name="src")

            kwargs = {"dim": dim}
            if start is not None:
                kwargs["start"] = start
            if end is not None:
                kwargs["end"] = end
            kwargs["step"] = step

            test_cases.append(
                TestCase(
                    inputs=[input_spec, src_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="slice_scatter - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SliceScatter")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, input, src, dim=0, start=None, end=None, step=1):
        return torch.slice_scatter(input, src, dim=dim, start=start, end=end, step=step)

    def infinicore_operator(self, input, src, dim=0, start=None, end=None, step=1):
        return infinicore.slice_scatter(input, src, dim=dim, start=start, end=end, step=step)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
