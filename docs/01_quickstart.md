# Quickstart

## Start Here

The public experiment names are:
- `quick`
- `main`
- `pyramidkernel_paper`
- `paper` (legacy compatibility preset)

Recommended order:
1. Try `quick` first
2. Use `main` for regular experiments
3. Use `pyramidkernel_paper` to reproduce the published experiment
4. Use `paper` only when you need the legacy STARK-compatible setting

## Example Commands

Run a batch in a stable way:

```bash
KERNELWEAVER_CONDA_ENV=stark \
bash scripts/run_batch.sh --detach --experiment main --output-dir runs/main
```

Run one task:

```bash
KERNELWEAVER_CONDA_ENV=stark \
bash scripts/run_single.sh --experiment main --level 1 --problem-id 25 --output-dir runs/l1_p25
```

Run a demo:

```bash
python stark_cli.py run-demo --task square_list --experiment quick --output-dir runs/demo_square_list
```

Watch one detached batch:

```bash
tail -f runs/main/launcher.log
```
