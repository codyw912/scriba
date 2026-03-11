# Benchmark Repository Boundary

This document defines what belongs in the public upstream repository versus what
should stay local-only during benchmark development.

## Public Upstream

Keep these in the repository:

- benchmark methodology docs:
  - `docs/benchmark_spec_v1.md`
  - `docs/benchmark_schema.md`
  - `docs/quality_evaluation_framework.md`
- reusable benchmark/report scripts,
- small synthetic fixture markdown and public contracts,
- benchmark manifests and directory scaffolding,
- profile examples and report-rendering logic.

These are part of the reproducible framework and should be understandable by an
external contributor without access to private benchmark data.

## Local/Internal Only

Do not treat these as routine public commits:

- generated run logs and reports:
  - `samples/matrix_runs.jsonl`
  - `samples/matrix_report.md`
  - `samples/matrix_report.json`
  - `samples/quick_runs.md`
  - `samples/quick_telemetry.md`
  - `samples/hosted_pareto.md`
- provider caches and downloaded model metadata,
- unpublished real-world paired/unpaired benchmark documents,
- ad-hoc experiment outputs, one-off calibration artifacts, and exploratory notes.

If a generated report or dataset slice is worth publishing, regenerate it from
the current framework and commit it intentionally as a fresh artifact.

## Practical Rule

Public upstream should answer:

- how benchmarking works,
- how to reproduce it,
- what schemas/manifests look like,
- what code computes the reports.

Local/internal benchmark work can include:

- sensitive or unreviewed inputs,
- historical experiment outputs,
- provider-specific cost/queue snapshots,
- anything not yet ready to defend as a public methodology artifact.

## Current Default

The repository defaults to keeping generated benchmark outputs local via
`.gitignore`, while preserving the synthetic benchmark framework in source
control.
