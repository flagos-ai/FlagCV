"""Verify add output, destination alias and input preservation."""

import pytest

from conftest import DummyReached, dummy
from evaluation.add import CASES, Case, make_inputs
from evaluation.reference import get_reference


@pytest.mark.add
@pytest.mark.parametrize(
    "case", (*CASES, Case((1, 3), "uint8")), ids=lambda case: case.id
)
def test_add(case, candidate, native_reference, pytestconfig):
    import torch

    device = pytestconfig.getoption("device")
    a, b, dst = make_inputs(case, device)
    if case.shape == (1, 3):
        a.copy_(a.new_tensor([[250, 1, 255]]))
        b.copy_(b.new_tensor([[10, 2, 255]]))

    if candidate is dummy:
        with pytest.raises(DummyReached):
            candidate(a, b, dst)
        return

    a_before, b_before = a.clone(), b.clone()
    reference = get_reference("add", device)
    expected = reference(a_before, b_before, torch.empty_like(dst))
    if case.shape == (1, 3):
        torch.testing.assert_close(expected, dst.new_tensor([[255, 3, 255]]))
    if native_reference is not None:
        native_dst = torch.empty_like(dst)
        native_reference(a_before.clone(), b_before.clone(), native_dst)
        torch.testing.assert_close(native_dst, expected)
        expected = native_dst

    assert candidate(a, b, dst) is dst
    torch.testing.assert_close(dst, expected)
    torch.testing.assert_close(a, a_before, rtol=0, atol=0)
    torch.testing.assert_close(b, b_before, rtol=0, atol=0)
