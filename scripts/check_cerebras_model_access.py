#!/usr/bin/env python3
"""Quick probe for Cerebras chat-completion model accessibility."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


DEFAULT_MODELS = [
    "zai-glm-4.7",
    "gpt-oss-120b",
    "qwen-3-235b-a22b-instruct-2507",
    "llama3.1-8b",
]


def _coerce_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()

    api_key = os.environ.get("CEREBRAS_API_KEY", "").strip()
    if not api_key:
        print("CEREBRAS_API_KEY is not set", file=sys.stderr)
        return 2

    try:
        from cerebras.cloud.sdk import Cerebras  # type: ignore[import-not-found]
    except ImportError:
        print(
            "Missing dependency: cerebras-cloud-sdk. Install with `uv add cerebras-cloud-sdk`.",
            file=sys.stderr,
        )
        return 2

    client = Cerebras(api_key=api_key)
    models = args.models or list(DEFAULT_MODELS)

    print("model,status,detail")
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                temperature=0.0,
                max_tokens=args.max_tokens,
            )
            payload = _coerce_dict(response)
            usage = payload.get("usage") if isinstance(payload, dict) else None
            completion_tokens = None
            if isinstance(usage, dict):
                completion_tokens = usage.get("completion_tokens")
            detail = (
                f"completion_tokens={completion_tokens}"
                if completion_tokens is not None
                else "ok"
            )
            print(f"{model},ok,{detail}")
        except Exception as exc:  # pragma: no cover - depends on remote API
            text = str(exc).replace("\n", " ").replace(",", ";")
            print(f"{model},error,{text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
