# FlagCV

## About

FlagCV is part of FlagOS, a fully open-source system software stack designed to unify the model–system–chip layers and foster an open and collaborative ecosystem. It enables a "develop once, run anywhere" workflow across diverse AI accelerators, unlocking hardware performance, eliminating fragmentation among AI chipset-specific software stacks, and substantially lowering the cost of porting and maintaining AI workloads.

FlagCV is a high-performance computer vision library implemented in Triton language. It provides optimized implementations of common image processing primitives (color conversion, filtering, geometric transforms, and arithmetic operations) behind OpenCV-shaped interfaces, benchmarked against OpenCV CUDA baselines to deliver reference-level correctness with competitive performance across diverse GPU architectures.

Built on FlagTree (a FlagOS-maintained Triton fork supporting multiple hardware backends), FlagCV offers a vendor-agnostic operator interface with pluggable backend support.
