"""FlagCV numerical comparison rules for Python operator tests."""


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
    torch.testing.assert_close(actual, expected, rtol=0, atol=1)


def assert_relative_close(actual, expected):
    """Apply the FlagCV relative threshold for the output dtype."""
    import torch

    rtol = {torch.float32: 1e-5, torch.float16: 1e-2}[expected.dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0)
