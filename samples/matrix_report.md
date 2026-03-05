# Matrix Report

- Generated: 2026-03-03T20:47:21.494201+00:00
- Matrix log: `/Users/cody/dev/scriba/samples/matrix_runs.jsonl`
- Artifacts root: `/Users/cody/dev/scriba/artifacts`

## Campaign Summary

| campaign_id | preset | rows | completed | failed | doctor_failed | skipped | avg_tok_s | avg_quality | started_at | last_at |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| cerebras-gpt-amortized-v2 | custom | 3 | 3 | 0 | 0 | 0 | 166.638 | 98.50 | 2026-03-03T20:47:05+00:00 | 2026-03-03T20:47:15+00:00 |

## Profile Summary

| profile | adapter | topology | provider | rows | completed | failed | doctor_failed | skipped | avg_tok_s | avg_visible_tok_s | avg_visible_tok_s_comparable | avg_completion_output_ratio | avg_quality | avg_content_f1 | avg_endpoint_recall | avg_contract_recall | hard_error_rate | contract_fail_rate | quality_gate_pass_rate | speed_gate_pass_rate | min_tok_s | max_tok_s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| profiles/remote/pipeline.profile.remote_cerebras_sdk_gpt_oss_120b.example.yaml | cerebras_sdk | remote | cerebras | 3 | 3 | 0 | 0 | 0 | 166.638 | 116.705 | 169.250 | 1.458 | 98.50 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.667 | 14.247 | 466.887 |

## Ranking

| rank | profile | topology | provider | completed | avg_tok_s |
|---:|---|---|---|---:|---:|
| 1 | profiles/remote/pipeline.profile.remote_cerebras_sdk_gpt_oss_120b.example.yaml | remote | cerebras | 3 | 166.638 |

## Visible Speed Ranking (Comparable Rows)

| rank | profile | topology | provider | completed | avg_visible_tok_s |
|---:|---|---|---|---:|---:|
| 1 | profiles/remote/pipeline.profile.remote_cerebras_sdk_gpt_oss_120b.example.yaml | remote | cerebras | 3 | 169.250 |

## Quality Ranking

| rank | profile | topology | provider | completed | avg_quality |
|---:|---|---|---|---:|---:|
| 1 | profiles/remote/pipeline.profile.remote_cerebras_sdk_gpt_oss_120b.example.yaml | remote | cerebras | 3 | 98.50 |

## Per Run

| timestamp | campaign_id | preset | profile | adapter | topology | provider | input | run_id | status | processed | tok_s | visible_tok_s | latency_s | completion_tokens | output_tokens_est | completion_output_ratio | reasoning_heavy | speed_gate_ok | quality | base_quality | content_f1 | endpoint_recall | endpoint_precision | heading_recall | heading_precision | contract_recall | contract_failures | quality_gate_ok | source | doctor_warning_count | doctor_warning_preview | validation_ok | hard_errors | missing_endpoints |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|---:|---:|
| 2026-03-03T20:47:05Z | cerebras-gpt-amortized-v2 | custom | profiles/remote/pipeline.profile.remote_cerebras_sdk_gpt_oss_120b.example.yaml | cerebras_sdk | remote | cerebras | mini_api.md | matrix-pipeline.profile.remote_cerebras_sdk_gpt_oss_120b.example-mini_api-20260303-154700 | completed | 1/1 | 18.779 | 11.614 | 4.047 | 76 | 47 | 1.617 | False | False | 95.5 | 95.5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | True | usage.completion_tokens | 0 | n/a | True | 0 | 0 |
| 2026-03-03T20:47:13Z | cerebras-gpt-amortized-v2 | custom | profiles/remote/pipeline.profile.remote_cerebras_sdk_gpt_oss_120b.example.yaml | cerebras_sdk | remote | cerebras | service_guide.md | matrix-pipeline.profile.remote_cerebras_sdk_gpt_oss_120b.example-service_guide-20260303-154706 | completed | 1/1 | 14.247 | 10.685 | 7.019 | 100 | 75 | 1.333 | False | True | 100.0 | 100.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | True | usage.completion_tokens | 0 | n/a | True | 0 | 0 |
| 2026-03-03T20:47:15Z | cerebras-gpt-amortized-v2 | custom | profiles/remote/pipeline.profile.remote_cerebras_sdk_gpt_oss_120b.example.yaml | cerebras_sdk | remote | cerebras | webhooks.md | matrix-pipeline.profile.remote_cerebras_sdk_gpt_oss_120b.example-webhooks-20260303-154713 | completed | 1/1 | 466.887 | 327.815 | 0.302 | 141 | 99 | 1.424 | False | True | 100.0 | 100.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | True | usage.completion_tokens | 0 | n/a | True | 0 | 0 |
