"""Live OpenCV CUDA baseline on caller-owned CUDA tensors and stream."""

import ctypes
import os
from functools import lru_cache
from operator import index
from pathlib import Path

_MODES = dict(add=0, subtract=1, multiply=2, bitwise_and=3, bitwise_not=4, flip=5)


@lru_cache(maxsize=None)
def _load_library(path):
    try:
        lib = ctypes.CDLL(path)
    except OSError as error:
        raise RuntimeError(
            "Cannot load the OpenCV CUDA bridge. Build adapters/opencv_cuda and "
            "set FLAGCV_OPENCV_LIBRARY to libflagcv_opencv_cuda.so"
        ) from error
    lib.flagcv_opencv_prepare.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.flagcv_opencv_prepare.restype = ctypes.c_void_p
    lib.flagcv_opencv_run.argtypes = [ctypes.c_void_p]
    lib.flagcv_opencv_run.restype = ctypes.c_int
    lib.flagcv_opencv_destroy.argtypes = [ctypes.c_void_p]
    lib.flagcv_opencv_destroy.restype = None
    lib.flagcv_opencv_error.restype = ctypes.c_char_p
    lib.flagcv_opencv_version.restype = ctypes.c_char_p
    return lib


class PreparedOperation:
    """GpuMat headers prepared once; no allocation or host copies in run()."""

    def __init__(self, op_name, args):
        import torch

        self.handle = None
        if op_name not in _MODES:
            raise ValueError(f"No OpenCV CUDA baseline for {op_name!r}")
        sources = 1 if op_name in ("bitwise_not", "flip") else 2
        expected_args = sources + 1 + (op_name == "flip")
        if len(args) != expected_args:
            raise ValueError(f"Invalid operands for {op_name}")
        tensors = args[: sources + 1]
        first = tensors[0]
        if first.device.type != "cuda" or torch.version.hip:
            raise ValueError("OpenCV CUDA baseline requires NVIDIA CUDA tensors")
        if first.dtype not in (torch.uint8, torch.float32):
            raise ValueError("OpenCV CUDA baseline requires uint8 or float32")
        if op_name.startswith("bitwise") and first.dtype != torch.uint8:
            raise ValueError("Bitwise baseline requires uint8")
        for tensor in tensors:
            if (
                tensor.device != first.device
                or tensor.dtype != first.dtype
                or tensor.ndim != 2
                or tensor.shape != first.shape
                or not tensor.is_contiguous()
                or tensor.numel() == 0
            ):
                raise ValueError(
                    "OpenCV baseline requires matching contiguous 2D tensors"
                )
        if max(first.shape) > 2**31 - 1:
            raise ValueError("OpenCV image dimensions must fit int32")
        self.args = args  # Keep every device allocation alive until close().
        self.dst = tensors[-1]
        self.device = first.device
        self.stream = torch.cuda.current_stream(self.device)
        flip_code = index(args[-1]) if op_name == "flip" else 0
        # Only the sign is significant; normalize before passing a C int.
        flip_code = (flip_code > 0) - (flip_code < 0)
        path = os.environ.get("FLAGCV_OPENCV_LIBRARY")
        if not path:
            path = str(
                Path(__file__).resolve().parents[1]
                / "build-opencv-cuda"
                / "libflagcv_opencv_cuda.so"
            )
        self.lib = _load_library(str(Path(path).resolve()))
        self.version = self.lib.flagcv_opencv_version().decode()
        with torch.cuda.device(self.device):
            self.handle = self.lib.flagcv_opencv_prepare(
                _MODES[op_name],
                *first.shape,
                0 if first.dtype == torch.uint8 else 5,
                first.data_ptr(),
                tensors[1].data_ptr() if sources == 2 else None,
                self.dst.data_ptr(),
                flip_code,
                self.stream.cuda_stream,
                self.device.index,
            )
        if not self.handle:
            raise RuntimeError(self.lib.flagcv_opencv_error().decode())

    def __call__(self, *args):
        # Prepared args/stream stay fixed for the duration of this benchmark case.
        if not self.handle:
            raise RuntimeError("OpenCV operation is closed")
        if self.lib.flagcv_opencv_run(self.handle):
            raise RuntimeError(self.lib.flagcv_opencv_error().decode())
        return self.dst

    def close(self):
        if self.handle:
            self.lib.flagcv_opencv_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def prepare(op_name, args):
    return PreparedOperation(op_name, args)
