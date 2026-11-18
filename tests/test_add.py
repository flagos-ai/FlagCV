"""Verify add output, destination alias and input preservation."""

from functools import partial

import pytest

from helpers.correctness import poison_output
from helpers.tolerance import (
    assert_bitwise_equal,
    assert_exact,
    assert_one_lsb,
    assert_relative_close,
)
from reference import get_reference
from workloads.add import CORRECTNESS_CASES, SATURATION_CASE, case_id, make_inputs


@pytest.mark.add
@pytest.mark.parametrize("case", CORRECTNESS_CASES, ids=case_id)
def test_add(case, candidate, native_reference, pytestconfig):
    import torch

    device = pytestconfig.getoption("device")
    inputs = partial(make_inputs, case, device)

    if candidate.source == "dummy":
        pytestconfig._evaluation.run_probe(
            candidate, inputs, pytestconfig._evaluation.active
        )
        return

    a, b, dst = inputs()
    a_before, b_before = a.clone(), b.clone()
    _, dtype, _ = case
    assert_output = assert_one_lsb if dtype == "uint8" else assert_relative_close
    reference = get_reference("add", device)
    expected = reference(a_before, b_before, torch.empty_like(dst))
    if case == SATURATION_CASE:
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
