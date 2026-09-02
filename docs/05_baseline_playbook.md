# Baseline Playbook

## What This Repo Is

- `PyramidKernel` is the public research implementation
- `KernelWeaver` and `stark` are retained as historical compatibility names

## Suggested Workflow

1. Start with `quick` if you only want a smoke test
2. Use `main` for regular experiments
3. Use `pyramidkernel_paper` to reproduce the published experiment

## Practical Rule

When you run an experiment, decide these four things first:
- `tasks`
- `backend`
- `route`
- `profile`

Then only edit lower-level files if you really need to.
