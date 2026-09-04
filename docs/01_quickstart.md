# Quickstart

## Start Here

The supported experiment profiles are:

- `quick`
- `main`
- `pyramidkernel_paper`

Recommended order:

1. Try `quick` first
2. Use `main` for regular experiments
3. Use `pyramidkernel_paper` to reproduce the published experiment

## Example Commands

Run a batch in a stable way:

```bash
PYRAMIDKERNEL_CONDA_ENV=pyramidkernel \
bash scripts/run_batch.sh --detach --experiment main --output-dir runs/main
```

Run one task:

```bash
PYRAMIDKERNEL_CONDA_ENV=pyramidkernel \
bash scripts/run_single.sh --experiment main --level 1 --problem-id 25 --output-dir runs/l1_p25
```

Run a demo:

```bash
pyramidkernel run-demo --task square_list --experiment quick --output-dir runs/demo_square_list
```

Watch one detached batch:

```bash
tail -f runs/main/launcher.log
```
