# Session Handoff (2026-03-02)

## Current state

- Hosted low-cost campaign completed:
  - campaign id: `hosted-lowcost-v1`
  - rows: `18` (`16` completed, `2` failed)
- Reports already generated:
  - `samples/matrix_report.md`
  - `samples/matrix_report.json`
  - `samples/hosted_pareto.md`
- Hosted pricing cache present:
  - `samples/openrouter_models.json`
- Tests currently passing:
  - `uv run --with pytest --with pytest-asyncio python -m pytest`
  - result: `28 passed`

## Next-session objective

Run second hosted campaign including stronger baseline and compare frontier shift.

Recommended command:

```bash
bash scripts/quick_openrouter_bench.sh --campaign-id hosted-lowcost-v2 --max-runs 21 --include-baseline
```

## Profiles in the hosted set

- Low-cost lane:
  - `profiles/remote/pipeline.profile.remote_openai_qwen25_7b.example.yaml`
  - `profiles/remote/pipeline.profile.remote_openai_qwen35_flash.example.yaml`
  - `profiles/remote/pipeline.profile.remote_openai_qwen3_coder_next.example.yaml`
  - `profiles/remote/pipeline.profile.remote_openai_llama31_8b.example.yaml`
  - `profiles/remote/pipeline.profile.remote_openai_mistral_nemo.example.yaml`
  - `profiles/remote/pipeline.profile.remote_openai_gpt4o_mini.example.yaml`
- Baseline added by `--include-baseline`:
  - `profiles/remote/pipeline.profile.remote_openai.example.yaml`

## Quick validation checks after v2 run

1. Confirm campaign rows in `samples/matrix_runs.jsonl` use `campaign_id=hosted-lowcost-v2`.
2. Check failures for rate-limit or validation regressions in `samples/matrix_report.md`.
3. Review frontier shifts in `samples/hosted_pareto.md`.
