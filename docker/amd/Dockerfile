# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e

# An isolated environment preserves the image's Torch and stock Triton files.
# FlagTree owns the imported triton module inside /opt/flagcv-env.
COPY flagtree-requirements.txt /tmp/flagcv-flagtree-requirements.txt
RUN python -m venv --system-site-packages /opt/flagcv-env \
    && /opt/flagcv-env/bin/python -m pip install --no-cache-dir --no-deps \
       --require-hashes -r /tmp/flagcv-flagtree-requirements.txt \
    && /opt/flagcv-env/bin/python -m pip install --no-cache-dir --no-deps cmake==3.31.10 \
    && rm /tmp/flagcv-flagtree-requirements.txt

# Python does not inherit a parent virtualenv's packages through
# --system-site-packages. Expose the pinned image's Torch/dependencies after
# this environment's FlagTree, without copying or reinstalling them.
RUN python -c 'import sysconfig; print(sysconfig.get_path("purelib"))' \
      > /opt/flagcv-env/lib/python3.12/site-packages/flagcv-base.pth \
    && /opt/flagcv-env/bin/python -c 'import torch, triton; print(torch.__version__, triton.__version__)'

ENV PATH="/opt/flagcv-env/bin:${PATH}"
ENV CMAKE_PREFIX_PATH="/opt/rocm"
WORKDIR /workspace
