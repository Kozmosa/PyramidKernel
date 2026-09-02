# Contributing

Thank you for helping improve PyramidKernel.

## Development setup

Use Python 3.10 or newer. Create a clean environment, install the development
dependencies, and run the unit tests before opening a pull request:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

GPU and model-provider tests are opt-in. They require the corresponding CUDA
toolchain, benchmark checkout, model weights, and API credentials.

## Pull requests

Keep pull requests focused and explain any change to evaluation, timing,
randomness, model routing, or benchmark handling. Do not commit `.env` files,
API responses containing secrets, generated run directories, or machine-local
paths. Add or update tests for behavior changes.

## Reproducibility

Record the repository commit, benchmark commit, configuration files, model
identifiers, hardware, CUDA/PyTorch versions, and timing settings for new
results. Do not present a result as paper-reproducible without this metadata.
