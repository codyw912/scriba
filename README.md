# scriba

CLI-first, backend-agnostic document normalization pipeline and evaluation kit.

## Why this repo exists

`scriba` is a clean-room split focused on local-file pipelines and model
evaluation loops.

- Keep: pipeline orchestration, profile-driven backends, resumable artifacts,
  validation, and telemetry.
- Drop (for now): web-service routes, URL rewriting, downloader/cache/image
  serving product surfaces.

## Scope (v0)

- Local file pipeline commands (`run`, `status`, `doctor`)
- Stage-oriented architecture and artifact state
- Quick fixture benchmarking and telemetry tracking

- Chunk sizing now supports model-aware defaults:
  - when `sectionize.target_tokens` and `sectionize.overlap_tokens` are omitted,
    values are inferred from model context metadata (explicit profile values still win)
  - sectionize compacts adjacent small chunks to better amortize request overhead
  - backend adapters provide model chunking hints (context length + sizing knobs);
    first-class adapters should source this from provider APIs/catalogs when available

## Non-goals (v0)

- Public HTTP service
- Hosted conversion API
- Opinionated lock-in to one model provider/runtime

## Install

```bash
uv sync --dev
```

## CLI

```bash
scriba run --profile profiles/pipeline.profile.example.yaml --input ./samples/file.pdf
scriba status --profile profiles/pipeline.profile.example.yaml --run-id run-20260301-120000
scriba doctor --profile profiles/pipeline.profile.example.yaml --input ./samples/file.pdf
```

## Profile examples

- `profiles/pipeline.profile.example.yaml` (no backend roles; map stage passthrough)
- `profiles/local_attached/pipeline.profile.local_attached_openai.example.yaml` (local service already running, e.g. LM Studio)
- `profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_openai.example.yaml` (recommended local_spawned default; scriba starts llama.cpp with Unsloth Qwen3.5-27B Q4_K_M)
- `profiles/local_spawned/pipeline.profile.local_spawned_openai.example.yaml` (alternative local_spawned runtime via mlx_lm.server)
- `profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_openai_highcap.example.yaml` (scriba starts llama.cpp higher-capacity Unsloth Qwen3.5-35B-A3B UD-Q4_K_XL)
- `profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_9b_bf16.example.yaml` (local_spawned Unsloth Qwen3.5-9B BF16 comparison)
- `profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_4b_bf16.example.yaml` (local_spawned Unsloth Qwen3.5-4B BF16 comparison)
- `profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_2b_bf16.example.yaml` (local_spawned Unsloth Qwen3.5-2B BF16 comparison)
- `profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_0p8b_bf16.example.yaml` (local_spawned Unsloth Qwen3.5-0.8B BF16 comparison)
- `profiles/local_attached/pipeline.profile.local_attached_openai_qwen35_35b_a3b_mxfp4.example.yaml` (local_attached LM Studio Qwen3.5-35B-A3B MLX MXFP4 comparison)
- `profiles/remote/pipeline.profile.remote_openai.example.yaml` (remote text backend + local-attached OCR backend)
- `profiles/hybrid/pipeline.profile.hybrid_local_spawned_ocr_remote_text.example.yaml` (local OCR backend + remote text normalization)
- `profiles/remote/pipeline.profile.remote_openai_qwen25_7b.example.yaml` (remote smaller-model comparison: Qwen2.5-7B)
- `profiles/remote/pipeline.profile.remote_openai_llama31_8b.example.yaml` (remote smaller-model comparison: Llama 3.1 8B)
- `profiles/remote/pipeline.profile.remote_openai_qwen35_flash.example.yaml` (remote Qwen3.5 Flash comparison)
- `profiles/remote/pipeline.profile.remote_openai_qwen3_next_80b_a3b.example.yaml` (remote Qwen3-Next-80B-A3B comparison)
- `profiles/remote/pipeline.profile.remote_openai_qwen3_coder_next.example.yaml` (remote Qwen3-Coder-Next comparison)
- `profiles/remote/pipeline.profile.remote_openai_gemini_2_5_flash.example.yaml` (production fast lane on OpenRouter)
- `profiles/remote/pipeline.profile.remote_openai_claude_sonnet_4_5.example.yaml` (production fallback lane on OpenRouter)
- `profiles/remote/pipeline.profile.remote_cerebras_sdk_llama31_8b.example.yaml` (Cerebras SDK profile: llama3.1-8b)
- `profiles/remote/pipeline.profile.remote_cerebras_sdk_gpt_oss_120b.example.yaml` (Cerebras SDK profile: gpt-oss-120b)
- `profiles/remote/pipeline.profile.remote_cerebras_sdk_qwen3_235b_a22b_instruct_2507.example.yaml` (Cerebras SDK profile: qwen-3-235b-a22b-instruct-2507)
- `profiles/remote/pipeline.profile.remote_cerebras_sdk_zai_glm_4_7.example.yaml` (Cerebras SDK profile: zai-glm-4.7)

| Profile | adapter | topology | provider | Typical use |
|---|---|---|---|---|
| `profiles/local_attached/pipeline.profile.local_attached_openai.example.yaml` | `openai_http` | `local_attached` | `lmstudio` | Reuse an already-running local server |
| `profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_openai.example.yaml` | `openai_http` | `local_spawned` | `llama_cpp` | Recommended local_spawned smoke/dev baseline |
| `profiles/local_spawned/pipeline.profile.local_spawned_openai.example.yaml` | `openai_http` | `local_spawned` | `mlx_lm` | Alternative local_spawned runtime comparison |
| `profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_openai_highcap.example.yaml` | `openai_http` | `local_spawned` | `llama_cpp` | Higher-capacity local_spawned baseline (heavier hardware) |
| `profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_9b_bf16.example.yaml` | `openai_http` | `local_spawned` | `llama_cpp` | Local BF16 Qwen3.5-9B comparison profile |
| `profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_4b_bf16.example.yaml` | `openai_http` | `local_spawned` | `llama_cpp` | Local BF16 Qwen3.5-4B comparison profile |
| `profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_2b_bf16.example.yaml` | `openai_http` | `local_spawned` | `llama_cpp` | Local BF16 Qwen3.5-2B comparison profile |
| `profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_0p8b_bf16.example.yaml` | `openai_http` | `local_spawned` | `llama_cpp` | Local BF16 Qwen3.5-0.8B comparison profile |
| `profiles/local_attached/pipeline.profile.local_attached_openai_qwen35_35b_a3b_mxfp4.example.yaml` | `openai_http` | `local_attached` | `lmstudio` | Local LM Studio MXFP4 Qwen3.5-35B-A3B profile |
| `profiles/remote/pipeline.profile.remote_openai.example.yaml` | `openai_http` | `remote` | `openrouter` | Use hosted text inference over network (with local OCR role available) |
| `profiles/remote/pipeline.profile.remote_openai_qwen25_7b.example.yaml` | `openai_http` | `remote` | `openrouter` | Smaller hosted comparison profile (Qwen2.5-7B) |
| `profiles/remote/pipeline.profile.remote_openai_llama31_8b.example.yaml` | `openai_http` | `remote` | `openrouter` | Smaller hosted comparison profile (Llama 3.1 8B) |
| `profiles/remote/pipeline.profile.remote_openai_qwen35_flash.example.yaml` | `openai_http` | `remote` | `openrouter` | Qwen3.5 Flash hosted comparison profile |
| `profiles/remote/pipeline.profile.remote_openai_qwen3_next_80b_a3b.example.yaml` | `openai_http` | `remote` | `openrouter` | Qwen3-Next-80B-A3B hosted comparison profile |
| `profiles/remote/pipeline.profile.remote_openai_qwen3_coder_next.example.yaml` | `openai_http` | `remote` | `openrouter` | Qwen3-Coder-Next hosted comparison profile |
| `profiles/remote/pipeline.profile.remote_openai_gemini_2_5_flash.example.yaml` | `openai_http` | `remote` | `openrouter` | Production fast lane |
| `profiles/remote/pipeline.profile.remote_openai_claude_sonnet_4_5.example.yaml` | `openai_http` | `remote` | `openrouter` | Production fallback lane |
| `profiles/remote/pipeline.profile.remote_cerebras_sdk_llama31_8b.example.yaml` | `cerebras_sdk` | `remote` | `cerebras` | Cerebras-hosted Llama 3.1 8B |
| `profiles/remote/pipeline.profile.remote_cerebras_sdk_gpt_oss_120b.example.yaml` | `cerebras_sdk` | `remote` | `cerebras` | Cerebras-hosted GPT-OSS 120B |
| `profiles/remote/pipeline.profile.remote_cerebras_sdk_qwen3_235b_a22b_instruct_2507.example.yaml` | `cerebras_sdk` | `remote` | `cerebras` | Cerebras-hosted Qwen 3 235B A22B |
| `profiles/remote/pipeline.profile.remote_cerebras_sdk_zai_glm_4_7.example.yaml` | `cerebras_sdk` | `remote` | `cerebras` | Cerebras-hosted Z.ai GLM 4.7 |

`llama_cpp` Unsloth notes:

- these profiles point `--model` to local Hugging Face cache GGUF paths
- if your snapshot hash/path differs, update the profile `command` model path
- expected memory footprint from Unsloth guidance:
  - Qwen3.5-27B Q4: around 17 GB total memory
  - Qwen3.5-35B-A3B 4-bit: around 22 GB total memory

## Backend topology

Backend configuration uses explicit axes so naming stays unambiguous:

- `adapter`: transport/integration method (`openai_http`, `cerebras_sdk`)
- `topology`:
  - `local_spawned`: scriba starts and stops the backend process
  - `local_attached`: backend already running locally (localhost/127.0.0.1)
  - `remote`: backend hosted elsewhere over network
- `provider`: runtime/vendor label (`mlx_lm`, `lmstudio`, `openrouter`, `openai`, etc.)
- `model_origin`: `local_weights`, `hosted_weights`, or `unknown`

This means a local model served by LM Studio is clearly represented as
`topology=local_attached` + `provider=lmstudio`, not remote.

Legacy compatibility: older backend `type` values (`local_openai`,
`external_openai`) are still parsed and mapped to canonical axes.

`cerebras_sdk` notes:

- requires Python package `cerebras-cloud-sdk`
- set `CEREBRAS_API_KEY` (for example in `.env`) for remote Cerebras profiles
- optional: set `SCRIBA_CEREBRAS_TIER=paygo` to use paygo context defaults
  (otherwise `free`-tier context defaults are used)

Map-stage worker/backoff controls:

- `stages.normalize_map.workers` controls shared worker count (default `1`)
- `SCRIBA_MAP_RATE_LIMIT_RETRIES` controls shared per-chunk retries after
  rate-limit responses (default `2`)
- tqdm progress bar for `normalize_map` appears on TTY stderr;
  disable with `SCRIBA_PROGRESS=0`

Role routing is independent from topology:

- `normalize_text` drives map-stage normalization
- `reduce_text` defaults to `normalize_text` unless explicitly overridden
- `ocr_vision` is reserved for OCR-specific model routing

Current runtime routing uses `normalize_text` for map-stage model calls; the
`reduce_text` fallback contract and `ocr_vision` role are now part of the
profile contract for upcoming OCR/reduce model routing updates.

Reasoning control (for OpenRouter-compatible models) can be configured per stage:

- `stages.normalize_map.reasoning_effort` (example: `none`)
- `stages.normalize_map.reasoning_exclude` (example: `true`)

## Quick evaluation

```bash
bash scripts/quick_eval.sh --doctor-only
bash scripts/quick_eval.sh
bash scripts/update_quick_eval.sh
```

Outputs:

- `samples/quick_runs.md`
- `samples/quick_telemetry.md`
- `docs/backend_decision_matrix.md` (history section auto-updated)

## Matrix runs

```bash
bash scripts/run_matrix.sh --doctor-only
bash scripts/run_matrix.sh --profile profiles/local_attached/pipeline.profile.local_attached_openai.example.yaml
bash scripts/run_matrix.sh --campaign-id one-model --profile profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_9b_bf16.example.yaml
bash scripts/run_matrix.sh --campaign-id qwen35-smalls --profile profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_4b_bf16.example.yaml --profile profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_2b_bf16.example.yaml --profile profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_0p8b_bf16.example.yaml --max-runs 9
bash scripts/run_matrix.sh --campaign-id qwen35-backend-compare --profile profiles/local_attached/pipeline.profile.local_attached_openai_qwen35_9b_bf16.example.yaml --profile profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_9b_bf16.example.yaml --profile profiles/local_attached/pipeline.profile.local_attached_openai_qwen35_4b_bf16.example.yaml --profile profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_4b_bf16.example.yaml --profile profiles/local_attached/pipeline.profile.local_attached_openai_qwen35_35b_a3b_mxfp4.example.yaml --profile profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_openai_highcap.example.yaml --max-runs 18
bash scripts/run_matrix.sh --preset fast-iterate --reset-log
scripts/render_matrix_report.py --json-output samples/matrix_report.json
```

Start a fresh matrix campaign log:

```bash
bash scripts/run_matrix.sh --reset-log --doctor-only
```

Remote profile smoke run (OpenRouter example):

```bash
# Add your key to `.env` (see `.env.example`) or export it in-shell.
bash scripts/run_matrix.sh --reset-log --profile profiles/remote/pipeline.profile.remote_openai.example.yaml --max-runs 1
scripts/render_matrix_report.py
```

Production-style two-lane run (fast + stronger fallback):

```bash
bash scripts/run_matrix.sh --reset-log --campaign-id hosted-production-lanes \
  --profile profiles/remote/pipeline.profile.remote_openai_gemini_2_5_flash.example.yaml \
  --profile profiles/remote/pipeline.profile.remote_openai_claude_sonnet_4_5.example.yaml \
  --max-runs 12
scripts/render_matrix_report.py
```

Cerebras direct-model quick run (SDK adapter):

```bash
# Add your key to `.env` (see `.env.example`) or export it in-shell.
bash scripts/quick_cerebras_bench.sh --campaign-id cerebras-direct-v1 --max-runs 12
```

Check Cerebras model access for your API key before running full campaigns:

```bash
uv run --env-file .env scripts/check_cerebras_model_access.py
```

Estimate large-run token budget locally (no model API calls):

```bash
# Run local extraction/clean first, then estimate budget from cleaned markdown.
uv run scripts/estimate_token_budget.py \
  --markdown artifacts/preflight-local/raw/cleaned.md \
  --model gpt-oss-120b
```

Guardrails in `scripts/run_matrix.sh` defaults:

- max pipeline runs: `6` (`--max-runs`)
- max sample size: `1500000` bytes (`--max-file-bytes`)
- stop profile after doctor failure (override with `--continue-on-doctor-fail`)
- optional clean campaign start: `--reset-log`

Presets for reproducible campaigns:

- `--preset fast-iterate`
  - profiles: `local_attached_openai` + recommended `local_spawned_llama_cpp_openai`
  - tuned for small-fixture iteration loops
- `--preset quality-check`
  - profiles: `local_attached_openai` + `local_spawned_llama_cpp_openai_highcap` + `remote_openai`
  - broader quality/sanity comparison pass

Campaign grouping/trends:

- each row in `samples/matrix_runs.jsonl` now includes `campaign_id` and `preset`
- `scripts/render_matrix_report.py` renders:
  - `Campaign Summary` (campaign-level rollups)
  - `Campaign Trend (Latest vs Previous)` with per-profile throughput deltas
  - quality metrics (`avg_quality`, quality ranking, per-run recall signals)

Outputs:

- `samples/matrix_runs.jsonl`
- `samples/matrix_report.md`
- `samples/matrix_report.json` (optional, with `--json-output`)
- schema reference: `docs/benchmark_schema.md`
- quality framework: `docs/quality_evaluation_framework.md`

## License

MIT. See `LICENSE`.
