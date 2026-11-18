"""Check the documented comparison boundaries for FlagCV operator tests."""

import pytest

from helpers.tolerance import (
    assert_bitwise_equal,
    assert_one_lsb,
    assert_relative_close,
)


def test_bitwise_equality_preserves_signed_zero():
    torch = pytest.importorskip("torch")
    negative_zero = torch.tensor([-0.0], dtype=torch.float32)
    assert_bitwise_equal(negative_zero, negative_zero.clone())
    with pytest.raises(AssertionError):
        assert_bitwise_equal(negative_zero, torch.zeros_like(negative_zero))


def test_one_lsb_accepts_one_level_without_wraparound():
    torch = pytest.importorskip("torch")
    expected = torch.tensor([0, 128, 255], dtype=torch.uint8)
    assert_one_lsb(torch.tensor([1, 127, 254], dtype=torch.uint8), expected)
    with pytest.raises(AssertionError):
        assert_one_lsb(torch.tensor([255, 128, 0], dtype=torch.uint8), expected)
    with pytest.raises(AssertionError):
        assert_one_lsb(torch.tensor([0, 130, 255], dtype=torch.uint8), expected)


@pytest.mark.parametrize(
    "dtype, within, outside",
    [("float32", 1.000008, 1.00002), ("float16", 1.008, 1.02)],
)
def test_relative_threshold_by_dtype(dtype, within, outside):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype)
    expected = torch.tensor([1.0, 0.0], dtype=dtype)
    assert_relative_close(torch.tensor([within, 0.0], dtype=dtype), expected)
    with pytest.raises(AssertionError):
        assert_relative_close(torch.tensor([outside, 0.0], dtype=dtype), expected)
    with pytest.raises(AssertionError):
        assert_relative_close(torch.tensor([1.0, 0.000001], dtype=dtype), expected)
