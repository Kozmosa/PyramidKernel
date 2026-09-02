# Paper results

These files are cleaned aggregate summaries from the paper experiment. They do
not contain raw model prompts, API responses, machine-local paths, or generated
candidate archives.

The aggregate was produced over 250 KernelBench tasks with a 10-candidate
budget. Correctness and Fast@1 use all tasks as the denominator; speedup
aggregates are computed over solved tasks. See `docs/reproducibility.md` for
the evaluation protocol and hardware caveats.

Raw run directories are intentionally not committed. They can contain secrets,
large generated sources, and paths specific to the original experiment host.
