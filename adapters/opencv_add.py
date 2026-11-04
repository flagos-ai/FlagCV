"""Expose the ops-baseline OpenCV CUDA recipe as a FlagCV evaluation callable."""

import os
import runpy
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_native(recipe):
    native = runpy.run_path(recipe)["native"]()
    if native is None:
        raise RuntimeError(
            "OpenCV CUDA bridge is unavailable. Check OPENCV_BASELINE_LIBRARY."
        )
    return native


def add(a, b, dst):
    """Run cv::cuda::add and return the caller's preallocated destination."""
    recipe = os.environ.get("OPENCV_BASELINE_RECIPE")
    if not recipe:
        raise RuntimeError("Set OPENCV_BASELINE_RECIPE to ops-baseline/ops/opencv/add.py")
    return _load_native(str(Path(recipe).resolve()))(a, b, dst)
