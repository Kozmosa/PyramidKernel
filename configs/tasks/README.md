# Task manifests

Task manifests are small, reviewable descriptions of the benchmark problems to
run. Use the canonical manifests for public experiments:

- `kb_smoke.yaml`: one task for a quick integration check;
- `main_l1_15.yaml`: the small development set;
- `kb_full_250.yaml`: all 250 KernelBench tasks used by the paper.

Historical experiment shards are intentionally excluded from this release;
only canonical manifests are part of the public reproduction workflow.

KernelBench itself is an external dependency. Pin its commit and follow its
license before downloading or redistributing benchmark files.
