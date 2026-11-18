"""Check the result and effects of an injected FlagCV candidate."""

from contextlib import nullcontext
from functools import partial

from helpers.tolerance import assert_bitwise_equal
from reference import get_reference


def poison_output(dst, expected):
    """Initialize the destination to values outside the allowed tolerance."""
    import torch

    # An inverted high bit is always 128 levels away, even at 127 and 128.
    if dst.dtype == torch.uint8:
        dst.copy_(expected ^ 0x80)
    elif dst.dtype == torch.float32:
        dst.view(torch.int32).copy_(~expected.view(torch.int32))
    else:
        dst.fill_(float("nan"))


def check(
    profile, case, candidate, native_reference, config, oracle, *, assert_output
):
    import torch

    device = config.getoption("device")
    build = partial(profile.make_inputs, case, device)
    state = config._evaluation
    if candidate.source == "dummy":
        state.run_probe(candidate, build, state.active)
        return

    args = build()
    extra = case.extra
    dst_index = len(profile.tensors) - 1
    sources = args[:dst_index]
    dst = args[dst_index]
    snapshots = tuple(source.clone() for source in sources)
    arrays = tuple(source.detach().cpu().numpy() for source in snapshots)
    expected = torch.from_numpy(oracle(*arrays, *extra)).to(dst.device)

    reference = get_reference(profile.name, device)
    reference_inputs = tuple(source.clone() for source in snapshots)
    reference_dst = torch.empty_like(dst)
    assert reference(*reference_inputs, reference_dst, *extra) is reference_dst
    assert_output(reference_dst, expected)
    for source, snapshot in zip(reference_inputs, snapshots):
        assert_bitwise_equal(source, snapshot)

    if native_reference is not None:
        native_inputs = tuple(source.clone() for source in snapshots)
        native_dst = torch.empty_like(dst)
        native_reference(*native_inputs, native_dst, *extra)
        assert_output(native_dst, expected)
        for source, snapshot in zip(native_inputs, snapshots):
            assert_bitwise_equal(source, snapshot)

    poison_output(dst, expected)
    scope = (
        torch.get_device_module(dst.device).device(dst.device)
        if dst.device.type != "cpu"
        else nullcontext()
    )
    with scope:
        assert candidate(*args) is dst
    assert_output(dst, expected)
    for source, snapshot in zip(sources, snapshots):
        assert_bitwise_equal(source, snapshot)
