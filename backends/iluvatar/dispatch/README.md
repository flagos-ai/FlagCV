# Iluvatar kernel selection

No platform kernel is registered yet, so selection always resolves to the
common registry and this directory holds no logic.

It becomes necessary as soon as a kernel is registered in `../kernels/`:
because platform registration is exclusive, any layout constraint that kernel
depends on must be verified here at build time, where the operand layout is
fully known.
