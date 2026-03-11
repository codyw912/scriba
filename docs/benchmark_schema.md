# Benchmark Output Schema

This document defines the minimum schema for repeatable profile/model
comparisons in `scribai` phase 5.

## Canonical backend axes

Backends are described using explicit axes:

- `adapter` (for example `litellm`)
- `topology` (`local_spawned`, `local_attached`, `remote`)
- `provider` (for example `mlx_lm`, `lmstudio`, `openrouter`)
- `model_origin` (`local_weights`, `hosted_weights`, `unknown`)

## Matrix run log (`samples/matrix_runs.jsonl`)

One JSON object per line.

Required fields:

- `timestamp` (string, UTC ISO-8601)
- `campaign_id` (string, campaign grouping id)
- `preset` (string, preset label or empty string for custom runs)
- `profile` (string, profile path used)
- `input` (string, input sample path)
- `status` (enum: `doctor_ok`, `doctor_failed`, `completed`, `completed_with_validation_errors`, `failed_runtime`, `skipped_large`, `skipped_limit`)

Conditional fields:

- `run_id` (string) when a pipeline run was attempted
- `file_bytes` and `max_file_bytes` for `skipped_large`
- `max_runs` for `skipped_limit`

## Telemetry source (`artifacts/<run_id>/map/manifest.json`)

Required manifest fields for comparison:

- `processed` (int)
- `chunk_count` (int)
- `processed_telemetry.requests` (int)
- `processed_telemetry.latency_s` (float)
- `processed_telemetry.output_tokens_est` (int)
- `processed_telemetry.effective_tokens_per_second` (float or null)

Optional preferred fields:

- `processed_telemetry.prompt_tokens` (int or null)
- `processed_telemetry.completion_tokens` (int or null)
- `processed_telemetry.total_tokens` (int or null)
- `processed_telemetry.chunks_with_usage` (int)

## Matrix report (`samples/matrix_report.md`)

Rendered from JSONL + artifact telemetry.

Campaign summary section columns:

- `campaign_id`
- `preset`
- `rows`
- `completed`
- `failed`
- `doctor_failed`
- `skipped`
- `avg_tok_s`
- `avg_quality`
- `started_at`
- `last_at`

Notes:

- `completed` summary counts include both `completed` and
  `completed_with_validation_errors` rows.
- `failed` summary counts represent `failed_runtime` rows.

Campaign trend section columns:

- `profile`
- `latest_avg_tok_s`
- `previous_avg_tok_s`
- `delta_tok_s`
- `delta_pct`

Profile summary section columns:

- `profile`
- `adapter`
- `topology`
- `provider`
- `rows`
- `completed`
- `failed`
- `doctor_failed`
- `skipped`
- `avg_tok_s`
- `avg_quality`
- `avg_endpoint_recall`
- `avg_contract_recall`
- `hard_error_rate`
- `contract_fail_rate`
- `min_tok_s`
- `max_tok_s`

Notes:

- `completed` summary counts include both `completed` and
  `completed_with_validation_errors` rows.
- `failed` summary counts represent `failed_runtime` rows.

Quality ranking section columns:

- `rank`
- `profile`
- `topology`
- `provider`
- `completed`
- `avg_quality`

Ranking section columns:

- `rank`
- `profile`
- `topology`
- `provider`
- `completed`
- `avg_tok_s`

Required table columns:

- `timestamp`
- `campaign_id`
- `preset`
- `profile`
- `adapter`
- `topology`
- `provider`
- `input`
- `run_id`
- `status`
- `processed`
- `tok_s`
- `quality`
- `endpoint_recall`
- `heading_recall`
- `contract_recall`
- `contract_failures`
- `source`
- `doctor_warning_count`
- `doctor_warning_preview`
- `validation_ok`
- `hard_errors`
- `missing_endpoints`

Benchmark metadata columns when the input belongs to `samples/benchmarks/v1/`:

- `fixture_id`
- `variant_id`
- `variant_family`
- `noise_level`
- `source_kind`

Optional quality/speed diagnostics columns:

- `base_quality`
- `content_f1`
- `endpoint_precision`
- `heading_precision`
- `quality_gate_ok`
- `latency_s`
- `completion_tokens`
- `output_tokens_est`
- `visible_tok_s`
- `completion_output_ratio`
- `reasoning_heavy`
- `speed_gate_ok`

Where `source` is:

- `usage.completion_tokens` when completion usage exists
- `output_tokens_est` when usage is absent

## Optional JSON summary (`samples/matrix_report.json`)

Generated with `scripts/render_matrix_report.py --json-output ...`.

Top-level fields:

- `generated_at`
- `matrix_log`
- `artifacts_root`
- `campaign_summary` (array)
- `profile_summary` (array)
- `ranking` (array)
- `trend_latest_vs_previous` (array)
- `benchmark_lane_summary` (array)
- `benchmark_lane_rows` (array)
- `rows` (array)

Benchmark lane row fields:

- `run_id`
- `fixture_id`
- `variant_id`
- `variant_family`
- `noise_level`
- `source_kind`
- `lane`
- `size_bucket`
- `doc_type`

Per-row optional quality fields:

- `quality_score` (float `0-100`)
- `quality_tier` (`excellent`, `good`, `fair`, `poor`)
- `endpoint_recall` (float `0-1`)
- `heading_recall` (float `0-1`)
- `contract_recall` (float `0-1` when fixture contract exists)
- `contract_failures` (int)

## Fixture contracts (`samples/contracts/*.json`)

Contract file path is derived from input fixture stem, for example:

- fixture: `samples/docs/mini_api.md`
- contract: `samples/contracts/mini_api.json`

Supported keys:

- `required_endpoints` (array of canonical `METHOD /path` strings)
- `required_headings` (array of heading titles, case-insensitive)
- `required_literals` (array of literals that must appear in output)
- `forbidden_literals` (array of literals that must not appear in output)
