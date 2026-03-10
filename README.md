# scriba

CLI-first pipeline for turning PDFs (and other OCR-able inputs) into clean,
accurate markdown.

The primary goal is reliability in real document workflows: predictable runs,
resumable artifacts, and profile-driven backend control.

## What it does

- Runs a deterministic stage pipeline on local inputs:
  - `extract -> clean -> sectionize -> normalize_map -> reduce -> validate -> export`
- Supports local, remote, and hybrid model backends through YAML profiles.
- Writes complete run artifacts under `~/.scriba/artifacts/<run_id>/...` by default for auditing and resume.
- Exposes simple CLI commands for run/status/doctor checks.

## Install

For source-tree development:

```bash
uv sync --group dev
```

Install as a local CLI tool:

```bash
uv tool install .
```

## Quick start

Run on a sample markdown fixture:

```bash
uv run scriba run \
  --input samples/docs/mini_api.md
```

For an installed tool, the equivalent command is just:

```bash
scriba run --input /path/to/file.pdf
```

By default, `scriba` uses the built-in `auto` preset when neither `--profile`
nor `--preset` is provided. `auto` picks the first configured provider API key
from `OPENROUTER_API_KEY`, `CEREBRAS_API_KEY`, or `OPENAI_API_KEY`.

If no provider key is set, use `--preset passthrough` (no model normalization)
or pass an explicit `--profile`.

Installed CLI usage is runtime-native: `scriba` stores optional user config in
`~/.scriba/config.yaml` and writes artifacts to `~/.scriba/artifacts/` by
default. Set `SCRIBA_HOME` to move both locations.

Run with a built-in preset (no profile path needed):

```bash
uv run scriba run \
  --preset openrouter \
  --input /path/to/file.pdf
```

Optional quick overrides:

- `--text-model <model_id>`
- `--ocr-model <model_id>`
- `--artifacts-root <path>`
- `--output <dir>` copies the final exported outputs to a user-facing directory

Validate profile + input before running:

```bash
uv run scriba doctor \
  --profile profiles/pipeline.profile.example.yaml \
  --input samples/docs/mini_api.md
```

Inspect a run by ID:

```bash
uv run scriba status \
  --profile profiles/pipeline.profile.example.yaml \
  --run-id <run_id>
```

## CLI

- `scriba run --profile ... --input ... [--run-id ...] [--resume]`
- `scriba run --input ...` (defaults to `--preset auto`)
- `scriba run --preset <auto|openrouter|cerebras|openai|passthrough> --input ...`
- `scriba run ... --output <dir>` copies `artifacts/<run_id>/final/` to `<dir>`
- `scriba status --profile ... --run-id ...`
- `scriba status --run-id ...` (defaults to `--preset auto`)
- `scriba status --preset <auto|openrouter|cerebras|openai|passthrough> --run-id ...`
- `scriba doctor --profile ... --input ...`
- `scriba doctor --input ...` (defaults to `--preset auto`)
- `scriba doctor --preset <auto|openrouter|cerebras|openai|passthrough> --input ...`

## Installed Usage

- Default home: `~/.scriba`
- Optional config: `~/.scriba/config.yaml`
- Default artifacts root: `~/.scriba/artifacts`
- Override home root with `SCRIBA_HOME=/custom/path`

Minimal optional config example:

```yaml
version: 1
defaults:
  preset: auto
  artifacts_root: ~/.scriba/artifacts
  provider_priority:
    - openrouter
    - cerebras
    - openai
models:
  openrouter: qwen/qwen3.5-35b-a3b
  cerebras: gpt-oss-120b
  openai: gpt-4o-mini
```

Precedence is: CLI flags > explicit `--profile` > `~/.scriba/config.yaml` > built-in defaults.

## Profiles

Profile files live in `profiles/` and are organized by topology:

- `profiles/pipeline.profile.example.yaml` - minimal baseline
- `profiles/local_spawned/` - scriba launches backend process
- `profiles/local_attached/` - connect to an already-running local backend
- `profiles/remote/` - hosted provider profiles
- `profiles/hybrid/` - mixed local/remote profile patterns

See `profiles/README.md` for layout details.

Those example profiles are primarily for source-tree and advanced custom usage.
The installed CLI does not depend on the repository `profiles/` directory for
default runs.

## OCR behavior (explicit)

For PDF inputs, extraction follows this order:

1. If a profile defines an `ocr_vision` role, `scriba` calls that vision model for OCR extraction.
2. If no `ocr_vision` role exists (or OCR vision extraction fails), `scriba` falls back to local `pymupdf4llm` extraction.

Current hosted/hybrid examples are intentionally anchored to **GLM-OCR** as the
default OCR model (`provider: glm_ocr`, `model: glm-ocr`).

You can switch OCR models/backends by editing profile config:

- `backends.ocr_backend` (adapter/topology/base_url/auth)
- `roles.ocr_vision.model` (OCR model id)

For this initial public push, GLM-OCR is the recommended default path.

## Environment

Copy `.env.example` to `.env` and set credentials as needed:

- `OPENROUTER_API_KEY`
- `CEREBRAS_API_KEY`
- `OPENAI_API_KEY`

Useful runtime controls:

- `SCRIBA_PROGRESS=0` disables tqdm map-stage progress bar
- `SCRIBA_MAP_RATE_LIMIT_RETRIES=<int>` sets per-chunk retry budget for rate-limit events
- `SCRIBA_CEREBRAS_TIER=paygo` switches Cerebras metadata assumptions to paygo tier
- `SCRIBA_BACKEND_PASSTHROUGH_LOGS=1` shows spawned backend stdout/stderr

## Reliability notes

- Use explicit `--run-id` for long jobs so reruns can safely `--resume`.
- Review run artifacts in `~/.scriba/artifacts/<run_id>/` by default (map telemetry, validation report, final markdown).
- Keep profile config as the source of truth for backend behavior (timeouts, workers, output limits).

## Large-document preflight

Estimate token budget locally from cleaned markdown before expensive remote runs:

```bash
uv run scripts/estimate_token_budget.py \
  --markdown artifacts/<preflight_run_id>/raw/cleaned.md \
  --model gpt-oss-120b
```

## Optional: model selection and benchmarking

Benchmarking exists to improve backend/model choices for the pipeline; it is not
required for normal usage.

- Sample benchmark command index: `samples/README.md`
- Benchmark schema: `docs/benchmark_schema.md`
- Quality framework: `docs/quality_evaluation_framework.md`
- Synthetic benchmark spec: `docs/benchmark_spec_v1.md`

## Contributing

See `CONTRIBUTING.md` for branch strategy, PR flow, and merge expectations.
See `RELEASING.md` for the release checklist and PyPI publish flow.
Basic GitHub Actions CI now validates tests, packaging, and installed-wheel smoke checks.

## License

MIT. See `LICENSE`.
