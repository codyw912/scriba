# Quality Evaluation Framework

This framework adds practical, repeatable quality scoring for model/profile
comparisons. It is intentionally heuristic-driven (not "perfect truth") and is
designed for fast iteration when comparing model families, quant levels, and
providers.

## Goals

- Catch factual/structural regressions that speed-only benchmarks miss.
- Enable side-by-side speed + quality tradeoff analysis in matrix campaigns.
- Keep scoring deterministic and cheap to compute from existing artifacts.

## Inputs per run

- Source text: `artifacts/<run_id>/raw/cleaned.md` (fallback: `raw/extracted.md`)
- Output text: `artifacts/<run_id>/final/merged.md`
- Validation report: `artifacts/<run_id>/final/validation_report.json`
- Optional fixture contract: `samples/contracts/<fixture_stem>.json`

Recommended fixture surfaces:

- smoke/minimal: `samples/docs/`
- expanded differentiation: `samples/docs_quality/`

## Quality metrics (v1)

- `endpoint_recall`
  - recall of canonical endpoint pairs (`METHOD /path`) from source to output
  - tolerant to markdown wrappers like ``GET `/v1/x``` and `**POST** `/v1/y``
- `heading_recall`
  - recall of markdown heading titles from source to output
- `endpoint_precision`
  - precision of canonical endpoint pairs in output vs source
- `heading_precision`
  - precision of markdown heading titles in output vs source
- `content_f1`
  - weighted F1 blend to better separate over-generated outputs:
    - `70%` endpoint F1
    - `30%` heading F1
- `path_recall`
  - recall of path-like tokens (for broader URI/path preservation signal)
- `number_recall`
  - recall of numeric tokens
- `length_ratio`
  - output/source character ratio, converted into a bounded `length_score`
- `contract_recall` (when contract exists)
  - recall across explicit fixture assertions:
    - required endpoints
    - required headings
    - required literals
    - forbidden literals

## Aggregate score

`quality_score` is `0-100`:

- `45%` endpoint recall
- `20%` heading recall
- `15%` path recall
- `5%` number recall
- `15%` length score

If run validation has hard errors, aggregate score is multiplied by `0.5`.

If fixture contract exists, final displayed quality is blended as:

- `final_quality = base_quality * contract_recall`

Additionally, report rows include a pragmatic quality gate:

- `quality_gate_ok = validation_ok && hard_errors==0 && quality>=65 && endpoint_recall>=0.9 && endpoint_precision>=0.9`

Quality tiers:

- `excellent` >= 90
- `good` >= 75
- `fair` >= 60
- `poor` < 60

## Reporting integration

`scripts/render_matrix_report.py` now includes quality in:

- campaign summary (`avg_quality`)
- profile summary (`avg_quality`, `avg_endpoint_recall`, `avg_contract_recall`, `hard_error_rate`, `contract_fail_rate`)
- quality ranking table
- per-run table (`quality`, `endpoint_recall`, `heading_recall`, `contract_recall`, `contract_failures`)

Per-run table now also includes:

- `base_quality` (before contract blending)
- `content_f1`
- `char_error_rate`
- `word_error_rate`
- `code_block_integrity_score`
- `table_retention_score`
- `hallucination_rate`
- `endpoint_precision`
- `heading_precision`
- `quality_gate_ok`

Profile summary now also includes:

- `quality_gate_pass_rate`
- `avg_content_f1`

## Known limitations

- This is a proxy framework, not semantic theorem proving.
- Recall-heavy metrics may over-score outputs that preserve key tokens but still
  rewrite nuances.
- Headings can be noisy for very short docs.
- Remote runs with high hidden reasoning token usage need generation tuning so
  cost/speed comparisons remain meaningful.

## Next iterations

- Add lightweight LSP/linter post-processing lane to reduce formatting burden on
  smaller models while preserving facts.
- Add judge-style pairwise scoring only for ambiguous tie-breaks, not as primary
  metric.
- Add periodic drift fixtures (versioned docs with controlled edits) to stress
  regression sensitivity across model/provider updates.
