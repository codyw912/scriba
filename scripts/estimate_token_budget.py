#!/usr/bin/env python3
"""Estimate token budget and free-tier fit from cleaned markdown."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from scribai.token_count import estimate_token_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--markdown", required=True, help="Path to cleaned markdown file"
    )
    parser.add_argument("--model", default="gpt-oss-120b", help="Model id")
    parser.add_argument("--target-tokens", type=int, default=20000)
    parser.add_argument("--overlap-tokens", type=int, default=2200)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--daily-limit", type=int, default=1_000_000)
    args = parser.parse_args()

    markdown_path = Path(args.markdown).expanduser().resolve()
    if not markdown_path.exists() or not markdown_path.is_file():
        raise SystemExit(f"Markdown file not found: {markdown_path}")

    text = markdown_path.read_text(encoding="utf-8")
    estimate = estimate_token_count(text, model=args.model)

    step = max(1, args.target_tokens - args.overlap_tokens)
    chunks = max(1, math.ceil(max(0, estimate.count - args.overlap_tokens) / step))
    effective_input_tokens = estimate.count + max(0, chunks - 1) * args.overlap_tokens
    reserved_output_tokens = chunks * args.max_output_tokens
    projected_total = effective_input_tokens + reserved_output_tokens
    pct_daily = (
        (projected_total / args.daily_limit) * 100 if args.daily_limit > 0 else 0
    )

    print(f"markdown: {markdown_path}")
    print(f"model: {args.model}")
    print(
        "token_estimate: "
        f"{estimate.count} (method={estimate.method}, encoding={estimate.encoding or 'n/a'})"
    )
    print(f"target_tokens: {args.target_tokens}")
    print(f"overlap_tokens: {args.overlap_tokens}")
    print(f"estimated_chunks: {chunks}")
    print(f"effective_input_tokens: {effective_input_tokens}")
    print(f"reserved_output_tokens: {reserved_output_tokens}")
    print(f"projected_total_quota_tokens: {projected_total}")
    print(f"daily_limit_tokens: {args.daily_limit}")
    print(f"daily_limit_usage_pct: {pct_daily:.2f}%")
    print(
        "fits_daily_limit: "
        f"{'yes' if args.daily_limit <= 0 or projected_total <= args.daily_limit else 'no'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
