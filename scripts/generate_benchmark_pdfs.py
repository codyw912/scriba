#!/usr/bin/env -S uv run --python 3.12

"""Generate deterministic clean PDF fixtures for benchmark v1."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean benchmark PDFs from fixture markdown.",
    )
    parser.add_argument(
        "--root",
        default="samples/benchmarks/v1",
        help="Benchmark root directory",
    )
    parser.add_argument(
        "--fixtures-manifest",
        default="manifests/fixtures.json",
        help="Fixture manifest path relative to root",
    )
    parser.add_argument(
        "--variants-manifest",
        default="manifests/variants.jsonl",
        help="Variants JSONL path relative to root",
    )
    parser.add_argument(
        "--fixture-id",
        action="append",
        default=[],
        help="Generate only selected fixture IDs (repeatable)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PDFs",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()
    fixtures_path = root / args.fixtures_manifest
    variants_path = root / args.variants_manifest

    if not fixtures_path.exists():
        raise SystemExit(f"fixtures manifest not found: {fixtures_path}")

    fixtures = _load_fixtures(fixtures_path)
    if args.fixture_id:
        include = set(args.fixture_id)
        fixtures = [f for f in fixtures if f["fixture_id"] in include]

    if not fixtures:
        print("No fixtures selected; nothing to generate.")
        return 0

    generated_rows: list[dict[str, Any]] = []
    sync_playwright = _import_sync_playwright()
    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        renderer_version = (
            f"playwright={_playwright_version()};chromium={browser.version}"
        )
        try:
            for fixture in fixtures:
                row = _generate_clean_pdf(
                    root=root,
                    fixture=fixture,
                    browser=browser,
                    renderer_version=renderer_version,
                    force=args.force,
                )
                generated_rows.append(row)
                print(f"generated: {row['fixture_id']} -> {row['pdf_path']}")
        finally:
            browser.close()

    _upsert_variants_manifest(
        variants_path=variants_path,
        generated_rows=generated_rows,
    )
    print(f"updated variants manifest: {variants_path}")
    return 0


def _import_sync_playwright() -> Any:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'playwright'. Install dev deps first with:\n"
            "  uv sync --group dev"
        ) from exc
    return sync_playwright


def _playwright_version() -> str:
    try:
        return importlib.metadata.version("playwright")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _launch_browser(playwright: Any) -> Any:
    try:
        return playwright.chromium.launch(headless=True)
    except Exception as exc:
        raise SystemExit(
            "Failed to launch Chromium. Install browsers first with:\n"
            "  uv run playwright install chromium"
        ) from exc


def _load_fixtures(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"Invalid fixtures manifest (expected array): {path}")

    fixtures: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        fixture_id = row.get("fixture_id")
        source_markdown = row.get("source_markdown")
        if not isinstance(fixture_id, str) or not fixture_id.strip():
            continue
        if not isinstance(source_markdown, str) or not source_markdown.strip():
            continue
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "source_markdown": source_markdown,
                "seed": _fixture_seed(fixture_id),
            }
        )
    return fixtures


def _fixture_seed(fixture_id: str) -> int:
    digest = hashlib.sha256(fixture_id.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _generate_clean_pdf(
    *,
    root: Path,
    fixture: dict[str, Any],
    browser: Any,
    renderer_version: str,
    force: bool,
) -> dict[str, Any]:
    fixture_id = str(fixture["fixture_id"])
    source_rel = str(fixture["source_markdown"])
    source_path = root / source_rel
    if not source_path.exists():
        raise SystemExit(f"Missing fixture markdown: {source_path}")

    markdown_text = source_path.read_text(encoding="utf-8")
    html = _render_fixture_html(markdown_text, title=fixture_id)

    output_dir = root / "generated_pdfs" / fixture_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "clean_pdf.pdf"

    if output_path.exists() and not force:
        pass
    else:
        context = browser.new_context(locale="en-US", timezone_id="UTC")
        page = context.new_page()
        page.set_content(html, wait_until="load")
        page.emulate_media(media="screen")
        page.pdf(
            path=str(output_path),
            format="A4",
            print_background=True,
            margin={
                "top": "16mm",
                "right": "14mm",
                "bottom": "16mm",
                "left": "14mm",
            },
            display_header_footer=False,
            prefer_css_page_size=True,
        )
        context.close()

    return {
        "fixture_id": fixture_id,
        "variant_id": "clean_pdf",
        "variant_family": "clean_pdf",
        "pdf_path": str(output_path.relative_to(root)),
        "seed": int(fixture["seed"]),
        "renderer": "playwright_chromium_pdf",
        "renderer_version": renderer_version,
        "noise_level": "none",
        "transform_params": {
            "pdf_format": "A4",
            "margin_mm": {"top": 16, "right": 14, "bottom": 16, "left": 14},
            "print_background": True,
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _render_fixture_html(markdown_text: str, *, title: str) -> str:
    try:
        import markdown as markdown_lib  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'markdown'. Install dev deps first with:\n"
            "  uv sync --group dev"
        ) from exc

    body = markdown_lib.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    @page {{ size: A4; margin: 16mm 14mm; }}
    :root {{
      color-scheme: light;
      --text: #0f172a;
      --muted: #334155;
      --border: #cbd5e1;
      --bg: #ffffff;
      --code: #f8fafc;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Georgia", "Times New Roman", serif;
      line-height: 1.5;
      font-size: 12pt;
    }}
    main {{ width: 100%; }}
    h1, h2, h3, h4 {{
      page-break-after: avoid;
      break-after: avoid;
      color: #0b1324;
      line-height: 1.25;
    }}
    h1 {{ font-size: 20pt; margin: 0 0 10pt 0; }}
    h2 {{ font-size: 16pt; margin: 14pt 0 8pt 0; }}
    h3 {{ font-size: 13pt; margin: 12pt 0 6pt 0; }}
    p, ul, ol {{ margin: 0 0 8pt 0; }}
    code {{
      background: var(--code);
      padding: 1pt 3pt;
      border-radius: 3pt;
      font-family: "Menlo", "Monaco", monospace;
      font-size: 10pt;
    }}
    pre {{
      background: var(--code);
      border: 1px solid var(--border);
      padding: 10pt;
      border-radius: 6pt;
      overflow: hidden;
      white-space: pre-wrap;
      word-break: break-word;
      margin: 10pt 0;
      page-break-inside: avoid;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 10pt 0;
      table-layout: fixed;
      page-break-inside: avoid;
      font-size: 10pt;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 6pt;
      vertical-align: top;
      word-break: break-word;
    }}
    th {{ background: #f1f5f9; color: var(--muted); }}
    blockquote {{
      margin: 10pt 0;
      padding: 0 0 0 10pt;
      border-left: 3px solid var(--border);
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>{body}</main>
</body>
</html>
"""


def _upsert_variants_manifest(
    *,
    variants_path: Path,
    generated_rows: list[dict[str, Any]],
) -> None:
    variants_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _load_variants_jsonl(variants_path)
    replacement_keys = {
        (row["fixture_id"], row["variant_id"]) for row in generated_rows
    }

    filtered = [
        row
        for row in existing_rows
        if (row.get("fixture_id"), row.get("variant_id")) not in replacement_keys
    ]
    filtered.extend(generated_rows)

    lines = [json.dumps(row, sort_keys=True) for row in filtered]
    content = "\n".join(lines)
    if content:
        content += "\n"
    variants_path.write_text(content, encoding="utf-8")


def _load_variants_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
