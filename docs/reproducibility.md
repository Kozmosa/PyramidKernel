# Reproducibility

## Paper configuration

The public preset is `configs/experiments/pyramidkernel_paper.yaml`. It uses the
complete `kb_full_250` manifest, the 10-attempt PyramidKernel search, the
`codeagent_claude` route, multi-model deliberation, and the isolated KernelBench
evaluator. Supply model identifiers and credentials through a private `.env`
file; never commit that file.

The paper reports two hardware contexts. The main PyramidKernel results use an
NVIDIA RTX 4090. The cross-system comparison uses an NVIDIA A100 80GB. Runtime,
driver, CUDA, PyTorch, and KernelBench versions must be recorded with
each run because GPU timings are not portable.

## Recommended protocol

1. Clone the repository and obtain the pinned KernelBench checkout.
2. Create a clean Python 3.10+ environment and install `.[cuda,dev]`.
3. Run the mock/demo smoke test and `python -m pytest -q`.
4. Run one KernelBench task before starting a batch.
5. Record the commit, benchmark commit, model IDs, configuration files, GPU,
   driver, CUDA/PyTorch versions, seed, timing method, and output checksum.
6. Revalidate the reported best candidates in a fresh process.

## Interpreting results

Correctness and Fast@1 use all tasks as the denominator. Speedup aggregates are
computed over solved tasks only. Report geometric mean and arithmetic mean
separately, and disclose any excessive-speedup filtering or outlier policy.
The evaluator executes generated code; follow `SECURITY.md` when running it.
