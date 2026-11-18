"""Check the result and effects of an injected FlagCV candidate."""

from contextlib import nullcontext
from functools import partial

from reference import get_reference


def check(
    profile, case, candidate, native_reference, config, oracle, *, exact_float=False
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
    compare_bits = exact_float and dst.dtype == torch.float32

    def assert_output(actual):
        if compare_bits:
            torch.testing.assert_close(
                actual.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0
            )
        else:
            tolerance = 1e-6 if dst.dtype == torch.float32 else 0
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def assert_source(actual, snapshot):
        if compare_bits:
            actual, snapshot = actual.view(torch.int32), snapshot.view(torch.int32)
        torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)

    reference = get_reference(profile.name, device)
    reference_inputs = tuple(source.clone() for source in snapshots)
    reference_dst = torch.empty_like(dst)
    assert reference(*reference_inputs, reference_dst, *extra) is reference_dst
    assert_output(reference_dst)
    for source, snapshot in zip(reference_inputs, snapshots):
        assert_source(source, snapshot)

    if native_reference is not None:
        native_inputs = tuple(source.clone() for source in snapshots)
        native_dst = torch.empty_like(dst)
        native_reference(*native_inputs, native_dst, *extra)
        assert_output(native_dst)
        for source, snapshot in zip(native_inputs, snapshots):
            assert_source(source, snapshot)

    # Poison every element so an omitted write differs from the expected result.
    if dst.dtype == torch.uint8:
        dst.copy_(255 - expected.to(torch.int16))
    elif compare_bits:
        dst.view(torch.int32).copy_(~expected.view(torch.int32))
    else:
        dst.fill_(float("nan"))
    scope = (
        torch.get_device_module(dst.device).device(dst.device)
        if dst.device.type != "cpu"
        else nullcontext()
    )
    with scope:
        assert candidate(*args) is dst
    assert_output(dst)
    for source, snapshot in zip(sources, snapshots):
        assert_source(source, snapshot)
