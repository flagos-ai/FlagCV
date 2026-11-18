"""FlagCV numerical comparison rules for Python operator tests."""

import json
from pathlib import Path

with Path(__file__).with_suffix(".json").open(encoding="utf-8") as policy_file:
    _TOLERANCE = json.load(policy_file)


def assert_exact(actual, expected):
    """Require exact element values."""
    import torch

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def assert_bitwise_equal(actual, expected):
    """Compare the underlying bytes, including the sign of floating-point zero."""
    import torch

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert_exact(
        actual.contiguous().reshape(-1).view(torch.uint8),
        expected.contiguous().reshape(-1).view(torch.uint8),
    )


def assert_one_lsb(actual, expected):
    """Allow one integer level of absolute error for uint8 outputs."""
    import torch

    assert actual.dtype == expected.dtype == torch.uint8
    torch.testing.assert_close(actual, expected, **_TOLERANCE["one_lsb"])


def assert_relative_close(actual, expected):
    """Apply the FlagCV relative threshold for the output dtype."""
    import torch

    dtype = str(expected.dtype).removeprefix("torch.")
    torch.testing.assert_close(actual, expected, **_TOLERANCE["relative"][dtype])
