# Samples

Place local fixture documents in `samples/docs/` for quick iteration runs.

Recommended loop:

1. Start with very small fixtures (3-5 pages)
2. Run `bash scripts/quick_eval.sh --doctor-only`
3. Run `bash scripts/quick_eval.sh`
4. Generate report table: `bash scripts/update_quick_eval.sh`

Generated outputs:

- `samples/quick_runs.md`
- `samples/quick_telemetry.md`
- `samples/hosted_pareto.md` (for hosted OpenRouter campaigns)

Decision guidance:

- `docs/backend_decision_matrix.md`

Matrix benchmarking outputs:

- `samples/matrix_runs.jsonl`
- `samples/matrix_report.md`
- `samples/matrix_report.json` (optional via `scripts/render_matrix_report.py --json-output ...`)
- `samples/hosted_pareto.md` (via `scripts/render_hosted_pareto.py`)

Benchmark v1 dataset scaffolding and PDF generation:

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
