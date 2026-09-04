# Third-party notices

The project uses external software and benchmark assets. Their licenses apply
separately from the Apache-2.0 license for this repository's own source code.

Before a release, record the exact commit and license for:

- KernelBench and any files obtained from its repository (upstream commit
  `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`, MIT licensed at the time of this
  release);
- PyTorch and its CUDA runtime packages;
- Optional Triton, TileLang, and NVIDIA CUTLASS/CuTe backend dependencies;
- model provider SDKs and any redistributed model outputs.

Do not vendor or redistribute an external benchmark, model weight, or provider
response until its license and terms permit that use.
