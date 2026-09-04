# Architecture

## Core Modules

- `pyramidkernel/core/loader.py`: thin KernelBench loader that reads official problems and builds generic `ModelNew` scaffolds
- `pyramidkernel/core/workflow.py`: staged search and refinement workflow
- `pyramidkernel/core/tree.py`: search-tree state and leaderboard tracking
- `pyramidkernel/core/context.py`: role-specific dynamic context construction

## Agent Modules

- `pyramidkernel/agents/plan_agent.py`: selects the optimization intent and grounded edit target
- `pyramidkernel/agents/code_agent.py`: implements the plan in the selected backend DSL
- `pyramidkernel/agents/debug_agent.py`: repairs compile, runtime, and correctness failures

## Provider Modules

- `pyramidkernel/providers/openai_provider.py`: OpenAI-compatible API provider
- `pyramidkernel/providers/claude_provider.py`: Claude-compatible API provider
- `pyramidkernel/providers/cudallm_provider.py`: local cudaLLM provider
- `pyramidkernel/providers/role_router.py`: per-role provider routing

## Evaluation Modules

- `pyramidkernel/evaluation/base.py`: shared evaluator helpers
- `pyramidkernel/evaluation/demo.py`: demo evaluator
- `pyramidkernel/evaluation/evaluator_paper.py`: official KernelBench-based evaluator
- `pyramidkernel/evaluation/validation.py`: saved-run revalidation

## Experiment Modules

- `pyramidkernel/experiment/batch_runner.py`: manifest loading and batch summaries
- `pyramidkernel/experiment/report_builder.py`: aggregate reports for completed runs

## Data Flow

1. `KernelBenchLoader` loads an official KernelBench problem into `TaskSpec`.
2. `workflow.py` initializes the root node from the generic `ModelNew` scaffold.
3. `PlanAgent` chooses the optimization strategy and structural anchor to edit.
4. `CodeAgent` emits a full updated Python module for the selected backend.
5. `KernelBenchPaperEvaluator` compiles, checks correctness, and measures runtime.
6. The search tree records candidates, failures, speedups, and the best node.
