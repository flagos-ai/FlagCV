"""Check the result and effects of an injected FlagCV candidate."""

from functools import partial

from reference import get_reference
from workloads.elementwise import make_inputs


def _oracle(op_name, sources, flip_code):
    import numpy as np

    arrays = [source.detach().cpu().numpy() for source in sources]
    if op_name == "flip":
        axes = (0,) if flip_code == 0 else (1,) if flip_code > 0 else (0, 1)
        # NumPy flip produces a view with negative strides.
        return np.flip(arrays[0], axis=axes).copy()
    if op_name == "bitwise_not":
        return np.bitwise_not(arrays[0])
    a, b = arrays
    if op_name == "bitwise_and":
        return np.bitwise_and(a, b)
    operation = np.subtract if op_name == "subtract" else np.multiply
    if a.dtype == np.uint8:
        return (
            operation(a.astype(np.int32), b.astype(np.int32))
            .clip(0, 255)
            .astype(np.uint8)
        )
    return operation(a, b)


def check(op_name, case, candidate, native_reference, config):
    import torch

    device = config.getoption("device")
    build = partial(make_inputs, op_name, case, device)
    state = config._evaluation
    if candidate.source == "dummy":
        state.run_probe(candidate, build, state.active)
        return

    args = build()
    extra = (case.flip_code,) if op_name == "flip" else ()
    dst_index = len(args) - 1 - len(extra)
    sources = args[:dst_index]
    dst = args[dst_index]
    snapshots = tuple(source.clone() for source in sources)
    expected = torch.from_numpy(_oracle(op_name, snapshots, case.flip_code)).to(
        dst.device
    )
    tolerance = {"rtol": 0, "atol": 0}
    if dst.dtype == torch.float32:
        tolerance = {"rtol": 1e-6, "atol": 1e-6}

    reference = get_reference(op_name, device)
    reference_inputs = tuple(source.clone() for source in snapshots)
    reference_dst = torch.empty_like(dst)
    assert reference(*reference_inputs, reference_dst, *extra) is reference_dst
    torch.testing.assert_close(reference_dst, expected, **tolerance)
    for source, snapshot in zip(reference_inputs, snapshots):
        torch.testing.assert_close(source, snapshot, rtol=0, atol=0)

    if native_reference is not None:
        native_inputs = tuple(source.clone() for source in snapshots)
        native_dst = torch.empty_like(dst)
        native_reference(*native_inputs, native_dst, *extra)
        torch.testing.assert_close(native_dst, expected, **tolerance)
        for source, snapshot in zip(native_inputs, snapshots):
            torch.testing.assert_close(source, snapshot, rtol=0, atol=0)

    # Poison every element so an omitted write differs from the expected result.
    if dst.dtype == torch.uint8:
        dst.copy_(255 - expected.to(torch.int16))
    else:
        dst.fill_(float("nan"))
    assert candidate(*args) is dst
    torch.testing.assert_close(dst, expected, **tolerance)
    for source, snapshot in zip(sources, snapshots):
        torch.testing.assert_close(source, snapshot, rtol=0, atol=0)
