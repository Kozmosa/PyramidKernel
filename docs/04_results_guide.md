# Results Guide

## Main Output Files

Each task run usually writes:

- `run.json`
- `best_code.py`
- `validation.json`

A batch run usually writes:

- `summary.json`
- `summary.csv`

## Important Fields

- `run_profile`: high-level preset name
- `search_profile`: resolved search config
- `evaluator_profile`: resolved evaluator config
- `measurement_profile`: resolved measurement config

## Report Command

```bash
pyramidkernel report-paper runs/main --output-dir runs/paper_report
```
