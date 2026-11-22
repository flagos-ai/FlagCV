"""Check bitwise_or values, destination aliasing, and input preservation."""

from contextlib import nullcontext
from functools import partial

import pytest
import torch
from helpers.tolerance import assert_bitwise_equal
from reference import get_reference
from workloads.bitwise_or import CORRECTNESS_CASES, OP_NAME, make_inputs


def _layout(tensor):
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.data_ptr(),
        tensor.shape,
        tensor.stride(),
        tensor.dtype,
        tensor.device,
    )


def _snapshot(tensor):
    if tensor is None:
        return None
    if tensor.dtype == torch.uint16:
        return tensor.view(torch.int16).clone().view(torch.uint16)
    return tensor.clone()


def _check_call(fn, args):
    snapshots = tuple(_snapshot(x) for x in args)
    layouts = tuple(_layout(x) if x is not None else None for x in args)
    dst = args[2]
    out = fn(*args)
    assert isinstance(out, torch.Tensor), "Candidate must return the supplied dst"
    assert _layout(out) == layouts[2]
    for i, original in enumerate(snapshots):
        if original is not None:
            assert _layout(args[i]) == layouts[i], "Input was rebound or resized"
            if i != 2:
                assert_bitwise_equal(args[i], original)
    if len(args) > 3 and args[3] is not None:
        unused = args[3] == 0
        assert_bitwise_equal(
            dst.view(torch.int16 if dst.dtype == torch.uint16 else dst.dtype)[unused],
            snapshots[2].view(torch.int16 if dst.dtype == torch.uint16 else dst.dtype)[
                unused
            ],
        )
    return dst


@pytest.mark.bitwise_or
@pytest.mark.parametrize("case", CORRECTNESS_CASES, ids=lambda case: case.id)
def test_bitwise_or(case, candidate, native_reference, pytestconfig):
    device = pytestconfig.getoption("device")
    build = partial(make_inputs, case, device)
    state = pytestconfig._evaluation
    if candidate.source == "dummy":
        state.run_probe(candidate, build, state.active)
        return

    args = build()
    expected = _check_call(get_reference(OP_NAME, device), build())
    scope = (
        torch.get_device_module(args[0].device).device(args[0].device)
        if args[0].device.type != "cpu"
        else nullcontext()
    )
    with scope:
        got = _check_call(candidate, args)
        if native_reference is not None:
            assert_bitwise_equal(got, _check_call(native_reference, build()))
    assert_bitwise_equal(got, expected)
