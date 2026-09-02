# PyramidKernel

PyramidKernel is a stage-structured, multi-agent framework for LLM-driven GPU
kernel optimization. It turns broad implementation hypotheses into
execution-grounded, bottleneck-focused refinement on KernelBench.

This repository is the public research implementation accompanying the paper.
The historical repository name is `KernelWeaver`; the `stark` package and CLI
remain available for compatibility, and `pyramidkernel` is the preferred
namespace and command name for new users.

## Status

- Paper: PyramidKernel: From Broad Hypotheses to Bottleneck-Focused Refinement
  for LLM-Driven GPU Kernel Optimization
- Release: research preview (`v0.1.0`)
- Scope: CUDA workflow, KernelBench evaluation, optional Triton/TileLang/CuTe
  backends, and mock/demo tests
- Results: see [`results/paper/`](results/paper/) for cleaned aggregate tables;
  reproduction instructions are in [`docs/reproducibility.md`](docs/reproducibility.md)

## What is included

The implementation contains the three shared stages from the paper:

1. S1 hypothesizes hardware-grounded optimization strategies;
2. S2 maintains a persistent candidate tree and measured leaderboard;
3. S3 realizes bounded region patches, evaluates them, and routes failures to
   repair.

The evaluator uses the official KernelBench correctness and timing path and can
run each candidate in an isolated subprocess. Generated code is still
untrusted; read [`SECURITY.md`](SECURITY.md) before running experiments.

## Requirements

- Linux with an NVIDIA GPU for KernelBench CUDA experiments;
- Python 3.10 or newer;
- a CUDA-compatible PyTorch installation;
- the external KernelBench checkout;
- model-provider credentials for non-mock runs;
- `nvcc` for CUDA extension builds (and `ncu` only for optional diagnostics).

The quick profile uses the mock provider and is intended for wiring checks. It
does not validate CUDA performance.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For a CUDA environment, install a PyTorch build matching the host driver, then
install the project with the optional extra:

```bash
python -m pip install -e ".[cuda,dev]"
```

Conda users may start from `environment.yml`; `environment.lock.yml` records a
Linux reference environment and is not a universal cross-platform lockfile.

Copy `.env.example` to `.env` and fill only the provider variables you intend
to use. Never commit `.env`, API responses, model weights, or credentials.

Set the external benchmark location in
`configs/runtime/kernelbench_paths.yaml`:

```yaml
kernelbench_root: /absolute/path/to/KernelBench
```

KernelBench is not redistributed by this repository. Obtain it from its
upstream project and follow its license and citation requirements.

## Quickstart

Run the unit tests and a provider-free demo first:

```bash
python -m pytest -q
python stark_cli.py run-demo --task square_list --experiment quick --output-dir runs/demo_square_list
```

Run one CUDA KernelBench task after configuring the benchmark and provider:

```bash
bash scripts/run_single.sh \
  --experiment pyramidkernel_paper \
  --level 1 \
  --problem-id 25 \
  --backend cuda \
  --route-config codeagent_claude \
  --output-dir runs/l1_p25
```

Use a fresh output directory for every run. `run.json`, `best_code.py`, and
validation artifacts are generated outputs and are ignored by default.

## Paper reproduction

Use the dedicated preset instead of guessing from development defaults:

```bash
bash scripts/run_batch.sh \
  --experiment pyramidkernel_paper \
  --output-dir runs/pyramidkernel_paper
```

The full protocol, hardware notes, metadata requirements, and metric
definitions are documented in [`docs/reproducibility.md`](docs/reproducibility.md).
The paper's main results were measured on an RTX 4090; cross-system comparisons
used an A100 80GB. GPU timing is hardware- and software-dependent, so exact
numbers cannot be expected on another machine.

## Repository layout

```text
src/                 Core workflow, agents, providers, evaluation, reporting
configs/             Public experiment, model, runtime, and task profiles
scripts/             Reproducible launch and validation helpers
docs/                Architecture, configuration, and reproduction guides
tests/               Unit and smoke tests
stark_cli.py         Compatibility CLI entry point
```

## Development

```bash
python -m pytest -q
python -m ruff check src tests
```

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

## License and citation

The original source code in this repository is released under the Apache
License 2.0. Third-party benchmark and library licenses apply separately; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The citation record is in [`CITATION.cff`](CITATION.cff); replace its draft
author and identifier fields with the final public paper metadata before
tagging the release.
