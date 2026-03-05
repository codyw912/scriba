# Benchmark Dataset v1

This directory contains benchmark fixtures and generated PDF variants for
deterministic OCR + normalization evaluation.

See `docs/benchmark_spec_v1.md` for the full spec.

## Quick start

Install project dev dependencies once:

```bash
uv sync --group dev
```

Scaffold missing directories/manifests (idempotent):

```bash
uv run scripts/scaffold_benchmark_v1.py
```

Optionally bootstrap initial fixtures from existing docs:

```bash
uv run scripts/scaffold_benchmark_v1.py --bootstrap-from "samples/docs/*.md"
```

Generate clean deterministic PDFs using Playwright Chromium:

```bash
uv run playwright install chromium
uv run scripts/generate_benchmark_pdfs.py
```

Generate stress/noise variants from clean PDFs:

```bash
uv run scripts/generate_benchmark_variants.py
```

Generate specific variant(s) for selected fixture(s):

```bash
uv run scripts/generate_benchmark_variants.py \
  --fixture-id mini_api \
  --variant scan_light \
  --variant scan_medium
```
