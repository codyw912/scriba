# Backend Decision Matrix

Use this as a practical profile chooser for pipeline runs. It is intentionally
backend-agnostic: pick what matches your constraints, then verify with telemetry.

## Quick picks

| Priority | Start here | Why | Watch-outs |
|---|---|---|---|
| Hosted low-cost baseline | `profiles/remote/pipeline.profile.remote_openai_qwen35_flash.example.yaml` | Very low token cost and broad context support on OpenRouter | Throughput is provider-dependent; rely on measured runs and Pareto output |
| Hosted low-cost quality candidate | `profiles/remote/pipeline.profile.remote_openai_qwen25_7b.example.yaml` | Competitive quality/cost on current normalization fixtures | May underperform on harder long-context documents |
| Hosted ultra-cheap floor | `profiles/remote/pipeline.profile.remote_openai_mistral_nemo.example.yaml` | Extremely low per-token cost; useful as cost anchor | Quality may trail stronger Qwen/DeepSeek classes |
| Fastest iteration | Local-attached or remote OpenAI-compatible profile | Attached/remote services can provide higher throughput and larger contexts | Confirm whether weights are local vs hosted for privacy requirements |
| Local-only development | Local-process profile | No external dependency, repeatable offline workflow | Throughput and startup costs vary by hardware |
| Mixed strategy | Local OCR + remote text model | Balances local control with faster normalization | More moving parts to configure and monitor |

## How to read telemetry

- `tok_s`: effective tokens per second for map stage.
- `source`: whether throughput uses provider completion tokens or estimated output.
- `usage_chunks`: chunks where provider usage metadata was available.
- `processed`: processed chunks over total chunks.
- `visible_tok_s`: estimated output tokens per second (`output_tokens_est / latency_s`).
- `completion_output_ratio`: `completion_tokens / output_tokens_est` (high values can indicate hidden reasoning/token inflation).

Interpretation notes:

- Prefer comparisons where `source=usage.completion_tokens`.
- If `source=output_tokens_est`, compare cautiously and treat as directional.
- If `processed` is less than total chunks, run was resumed or partial.
- If `completion_output_ratio` is very high, `tok_s` may overstate user-visible throughput; prefer `visible_tok_s` for practical speed comparisons.
- Very small outputs can be dominated by fixed latency/TTFT overhead; treat rows with low `output_tokens_est` as low-confidence for speed ranking.

Practical comparison rule:

- Use comparable rows where `output_tokens_est>=60` and `completion_output_ratio<=3`, then rank by `visible_tok_s`.

## Tracked quick-run results

Latest telemetry table:

- `samples/quick_telemetry.md`
- `samples/hosted_pareto.md`

Refresh and append history:

```bash
bash scripts/update_quick_eval.sh

# hosted cost/speed frontier (OpenRouter)
scripts/render_hosted_pareto.py --require-validation-ok
```

## Quick-run telemetry history

This section is auto-updated by `scripts/append_quick_telemetry_history.py`.
<!-- quick-telemetry-history:start -->

| generated_at | profile | runs | avg_tok_s | min_tok_s | max_tok_s |
|---|---|---:|---:|---:|---:|
| 2026-03-01T20:25:48.740974+00:00 | profiles/pipeline.profile.example.yaml | 2 | n/a | n/a | n/a |
| 2026-03-01T18:51:53.433747+00:00 | profiles/pipeline.profile.example.yaml | 2 | n/a | n/a | n/a |
| 2026-03-01T18:26:37.775256+00:00 | profiles/pipeline.profile.example.yaml | 2 | n/a | n/a | n/a |
| 2026-03-01T18:15:05.823299+00:00 | profiles/pipeline.profile.example.yaml | 2 | n/a | n/a | n/a |
| 2026-03-01T18:06:47.790831+00:00 | profiles/pipeline.profile.example.yaml | 2 | n/a | n/a | n/a |
| 2026-03-01T17:24:33.497956+00:00 | profiles/pipeline.profile.example.yaml | 2 | n/a | n/a | n/a |
| 2026-03-01T17:24:08.165573+00:00 | profiles/pipeline.profile.example.yaml | 2 | n/a | n/a | n/a |

<!-- quick-telemetry-history:end -->
