# Samples

Place local fixture documents in `samples/docs/` for quick iteration runs.

## Public Vs Local-Only Assets

Public upstream should contain:

- small synthetic fixtures and contracts that define the benchmark method,
- benchmark manifests and dataset scaffolding under `samples/benchmarks/v1/`,
- reusable scripts and schemas,
- documentation describing how to reproduce runs.

See `docs/benchmark_repository_boundary.md` for the full repository policy.

Local-only benchmark/testing outputs should not be committed:

- generated run logs and markdown reports,
- hosted Pareto snapshots and quick telemetry outputs,
- provider caches and ad-hoc calibration outputs,
- any real-world benchmark PDFs/markdown that are not explicitly cleared for public release.

The generated files listed below are intended as local working outputs and are
gitignored by default.

Recommended loop:

1. Start with very small fixtures (3-5 pages)
2. Run `bash scripts/quick_eval.sh --doctor-only`
3. Run `bash scripts/quick_eval.sh`
4. Generate report table: `bash scripts/update_quick_eval.sh`

Local generated outputs:

- `samples/quick_runs.md`
- `samples/quick_telemetry.md`
- `samples/hosted_pareto.md` (for hosted OpenRouter campaigns)

Decision guidance:

- `docs/backend_decision_matrix.md`

Local matrix benchmarking outputs:

- `samples/matrix_runs.jsonl`
- `samples/matrix_report.md`
- `samples/matrix_report.json` (optional via `scripts/render_matrix_report.py --json-output ...`)
- `samples/hosted_pareto.md` (via `scripts/render_hosted_pareto.py`)

Benchmark v1 public dataset scaffolding and PDF generation:

- spec: `docs/benchmark_spec_v1.md`
- root: `samples/benchmarks/v1/`
- setup (once): `uv sync --group dev`
- scaffold: `uv run scripts/scaffold_benchmark_v1.py`
- clean PDF generation: `uv run scripts/generate_benchmark_pdfs.py`
- variant generation: `uv run scripts/generate_benchmark_variants.py`

Hosted low-cost OpenRouter campaign (recommended):

```bash
bash scripts/quick_openrouter_bench.sh
```

Include stronger remote baseline in the same run:

```bash
bash scripts/quick_openrouter_bench.sh --include-baseline
```

`samples/matrix_report.md` now includes both speed and quality sections.

Fixture contracts for quality assertions:

- place per-fixture contract files in `samples/contracts/<fixture_stem>.json`
- supported keys:
  - `required_endpoints`
  - `required_headings`
  - `required_literals`
  - `forbidden_literals`

Preset matrix campaigns:

- Fast iteration: `bash scripts/run_matrix.sh --preset fast-iterate --reset-log`
- Broader quality check: `bash scripts/run_matrix.sh --preset quality-check --reset-log`

For the current public upstream boundary, keep `samples/benchmarks/v1/gold_*`
and manifests in-repo, but treat `generated_pdfs/`, `real_paired/`, and
`real_unpaired/` contents as publish-on-purpose assets rather than routine
commits.
