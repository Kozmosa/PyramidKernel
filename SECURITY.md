# Security Policy

## Scope

KernelWeaver/PyramidKernel generates and evaluates Python and CUDA programs.
Generated programs are untrusted code and must not be evaluated on a machine
that contains production credentials or sensitive data.

The subprocess evaluator provides timeouts and process cleanup, but it is not a
complete security sandbox. For experiments, use a disposable GPU host or a
container with a scrubbed environment, read-only mounts where possible, and no
network access.

## Reporting a vulnerability

Please do not open a public issue for a credential leak or an exploitable
execution issue. Contact the maintainers privately through the contact listed
in the repository profile and include a minimal reproduction, affected commit,
and impact. Never include live API keys in a report.

## Credential handling

Keep provider credentials in a local `.env` file or a secret manager. The
repository intentionally ships only `.env.example`. Rotate any credential that
may have appeared in logs, run artifacts, or command history.
