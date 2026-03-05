# Benchmark Spec v1 (Synthetic + Paired PDF Evaluation)

This spec defines a reproducible benchmark process for evaluating OCR +
normalization quality and speed across models/backends.

The core design is:

- start from known-good markdown fixtures,
- generate deterministic PDF variants with increasing difficulty,
- run the full pipeline,
- score outputs against known references.

This complements the existing matrix/report framework and keeps comparisons
scientific and repeatable.

## Objectives

- Measure OCR fidelity and normalization fidelity separately.
- Benchmark speed/quality/cost tradeoffs under controlled document noise.
- Keep runs reproducible (fixed variants, fixed seeds, fixed manifests).
- Preserve output artifacts even when validation errors occur.

## Canonical Renderer Decision

Use **Playwright (Chromium print-to-PDF)** as the canonical markdown->PDF
renderer for synthetic fixtures.

Rationale:

- deterministic in CI when browser/runtime is pinned,
- strong layout/CSS control (tables, code blocks, page breaks),
- realistic digital-PDF baseline before scan/noise transforms.

## Dataset Layout

Use this directory structure:

```text
samples/benchmarks/v1/
  gold_markdown/
    <fixture_id>.md
  gold_contracts/
    <fixture_id>.json
  generated_pdfs/
    <fixture_id>/<variant_id>.pdf
  manifests/
    fixtures.json
    variants.jsonl
  real_paired/
    pdf/<fixture_id>.pdf
    source_markdown/<fixture_id>.md
  real_unpaired/
    pdf/<fixture_id>.pdf
```

Notes:

- `gold_markdown` is source-of-truth for strict scoring.
- `gold_contracts` uses the existing contract schema style where possible.
- `real_paired` is for real-world documents that have known references.
- `real_unpaired` is robustness monitoring only (not strict full-text scoring).

## Fixture and Variant Manifests

### `fixtures.json` (array)

Required fields per fixture:

- `fixture_id`
- `source_markdown`
- `size_bucket` (`tiny`, `small`, `medium`, `large`, `xl`)
- `doc_type` (`api`, `guide`, `mixed`, ...)
- `has_contract` (bool)

Optional:

- `tags` (array)
- `notes`

### `variants.jsonl` (one JSON row per generated PDF)

Required fields:

- `fixture_id`
- `variant_id`
- `variant_family` (`clean_pdf`, `layout_light`, `scan_light`, `scan_medium`, `scan_hard`, `mixed_content`)
- `pdf_path`
- `seed`
- `renderer` (`playwright_chromium_pdf`)
- `renderer_version`
- `noise_level` (`none`, `low`, `medium`, `high`)
- `transform_params` (object)
- `generated_at` (UTC ISO-8601)

## Variant Taxonomy

Baseline and stress variants:

- `clean_pdf`: deterministic clean digital PDF.
- `layout_light`: typography and layout pressure without OCR corruption.
- `scan_light`: mild rasterization artifacts (compression/noise/skew).
- `scan_medium`: stronger artifacts with readability impact.
- `scan_hard`: heavy artifacts, mixed DPI/rotation/contrast loss.
- `mixed_content`: tables/code/list-heavy and structure-sensitive layout.

All variants must be seed-driven and reproducible.

## Evaluation Lanes

Evaluate each run in one or more lanes:

- `ocr_lane`:
  - compare `artifacts/<run_id>/raw/extracted.md` (or `raw/cleaned.md`) to gold markdown.
  - isolates OCR/extraction quality.
- `full_pipeline_lane`:
  - compare `artifacts/<run_id>/final/merged.md` to gold markdown.
  - captures end-to-end quality.
- `contract_lane`:
  - compare against explicit fixture contracts.
- `robustness_lane` (real-unpaired only):
  - structural consistency metrics only; no strict full-text truth scoring.

## Metrics

Keep current quality/speed metrics and add OCR-oriented metrics.

Existing metrics retained:

- endpoint/heading recall + precision,
- content_f1,
- path_recall, number_recall, length_score,
- contract_recall and contract_failures,
- tok/s, visible_tok/s, completion_output_ratio, latency.

New metrics to add:

- OCR normalized edit distance (and/or CER/WER),
- code-block integrity score,
- table retention score,
- hallucination rate (output-only endpoints/headings),
- omission severity buckets for critical entities.

## Runtime Status Semantics (Required Change)

Do not discard outputs on validation hard errors in live/paid benchmarking.

Introduce status semantics:

- `completed`
- `completed_with_validation_errors`
- `failed_runtime`

Strict fail mode may remain available for CI/profile checks, but should not be
the default for expensive live runs.

## Selection Workflow

Three-stage selection process:

1. **Screening** on `clean_pdf` + `scan_light` subset.
2. **Promotion** of Pareto candidates to full variant set.
3. **Decision** using constraints:
   - minimum quality floor,
   - max budget,
   - throughput target,
   - acceptable hard-error rate.

## Reporting Extensions

Extend matrix/report rows with benchmark metadata:

- `fixture_id`, `variant_id`, `variant_family`, `noise_level`, `lane`, `source_kind`.

Add aggregate cuts:

- by noise level,
- by lane,
- by fixture size bucket,
- by doc type.

Add variance/confidence reporting for queue-sensitive providers.

## Implementation Plan

Phase 1:

- scaffold dataset directories and manifest schemas,
- add deterministic markdown->PDF generation script.

Phase 2:

- add variant/noise generation script,
- generate v1 fixture set.

Phase 3:

- integrate lane-aware scoring into report generation,
- add OCR metrics and new row dimensions.

Phase 4:

- update runtime status semantics to preserve exports,
- wire `completed_with_validation_errors` through matrix/report.

Phase 5:

- add constrained selection summary (quality floor + budget + speed).

## Non-Goals (v1)

- No judge-LLM semantic grading as primary metric.
- No random/non-deterministic variant generation.
- No requirement for unpaired real docs to have strict full-text scores.
