"""Verify add output, destination alias and input preservation."""

import pytest

from helpers.correctness import poison_output
from helpers.tolerance import (
    assert_bitwise_equal,
    assert_exact,
    assert_one_lsb,
    assert_relative_close,
)
from reference import get_reference
from workloads.add import CASES, Case, make_inputs


@pytest.mark.add
@pytest.mark.parametrize(
    "case", (*CASES, Case((1, 3), "uint8")), ids=lambda case: case.id
)
def test_add(case, candidate, native_reference, pytestconfig):
    import torch

    device = pytestconfig.getoption("device")

    def inputs():
        a, b, dst = make_inputs(case, device)
        if case.shape == (1, 3):
            a.copy_(a.new_tensor([[250, 1, 255]]))
            b.copy_(b.new_tensor([[10, 2, 255]]))
        return a, b, dst

    if candidate.source == "dummy":
        pytestconfig._evaluation.run_probe(
            candidate, inputs, pytestconfig._evaluation.active
        )
        return

    a, b, dst = inputs()
    a_before, b_before = a.clone(), b.clone()
    assert_output = assert_one_lsb if case.dtype == "uint8" else assert_relative_close
    reference = get_reference("add", device)
    expected = reference(a_before, b_before, torch.empty_like(dst))
    if case.shape == (1, 3):
        assert_exact(expected, dst.new_tensor([[255, 3, 255]]))
    if native_reference is not None:
        native_dst = torch.empty_like(dst)
        native_reference(a_before.clone(), b_before.clone(), native_dst)
        assert_output(native_dst, expected)

    poison_output(dst, expected)
    assert candidate(a, b, dst) is dst
    assert_output(dst, expected)
    if native_reference is not None:
        assert_output(dst, native_dst)
    assert_bitwise_equal(a, a_before)
    assert_bitwise_equal(b, b_before)
