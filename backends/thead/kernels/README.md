# T-Head platform kernels

This registry is empty, so every operation resolves to `kernels/registry.json`
and runs the common Triton kernel. That is the intended starting point: the
common kernel is written to be correct on all platforms.

Add a kernel here only when measurement shows a worthwhile gain, and register
it in `registry.json` under provider `thead_triton`.

Registration is exclusive. Once an operation appears here, a failure does not
silently fall back to the common kernel, so any layout constraint the kernel
relies on has to be checked in `../dispatch/` at build time.
