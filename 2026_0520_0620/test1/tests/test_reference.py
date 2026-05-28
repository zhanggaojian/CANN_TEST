import math

import numpy as np
import torch
import torch.nn.functional as F


def celu_numpy(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return np.maximum(x, 0.0) + np.minimum(alpha * (np.exp(x / alpha) - 1.0), 0.0)


def assert_case(values, shape, alpha):
    arr = np.array(values, dtype=np.float32).reshape(shape)
    actual = celu_numpy(arr, alpha)
    expected = F.celu(torch.from_numpy(arr), alpha=alpha).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True)


def test_required_examples():
    assert_case([-1.0, 0.0, 1.0, 2.0], (4,), 1.0)
    assert_case([0.5, -0.5, 1.5, -1.5], (2, 2), 2.0)
    assert_case([-2.0], (1,), 0.5)

    arr = np.array([math.inf, -math.inf, math.nan], dtype=np.float32)
    actual = celu_numpy(arr, 1.0)
    assert math.isinf(actual[0]) and actual[0] > 0
    assert actual[1] == -1.0
    assert math.isnan(actual[2])


def test_shapes_and_alphas():
    rng = np.random.default_rng(20260528)
    cases = [
        ((), 1.0),
        ((1,), 0.1),
        ((33,), 0.5),
        ((7, 11), 2.0),
        ((3, 5, 9), 5.0),
        ((2, 3, 4, 17), 1.0),
        ((2, 1, 3, 5, 7), 2.0),
    ]
    for shape, alpha in cases:
        size = int(np.prod(shape)) if shape else 1
        values = rng.normal(loc=0.0, scale=8.0, size=size).astype(np.float32)
        assert_case(values, shape, alpha)


if __name__ == "__main__":
    test_required_examples()
    test_shapes_and_alphas()
    print("reference tests passed")

