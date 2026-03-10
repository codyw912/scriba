"""Tests for token counting helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from scribai.token_count import estimate_token_count, estimated_chars_for_tokens


class _FakeEncoding:
    def __init__(self, *, name: str, n_tokens: int) -> None:
        self.name = name
        self._n_tokens = n_tokens

    def encode_ordinary(self, text: str) -> list[int]:
        del text
        return list(range(self._n_tokens))


def test_estimate_token_count_prefers_model_encoding(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        encoding_for_model=lambda model: _FakeEncoding(name=f"enc:{model}", n_tokens=7),
        get_encoding=lambda name: _FakeEncoding(name=name, n_tokens=3),
    )
    monkeypatch.setitem(sys.modules, "tiktoken", fake_module)

    result = estimate_token_count("hello world", model="gpt-oss-120b")
    assert result.count == 7
    assert result.method == "tiktoken_model"
    assert result.encoding == "enc:gpt-oss-120b"


def test_estimate_token_count_falls_back_to_heuristic_when_unavailable(
    monkeypatch,
) -> None:
    fake_module = SimpleNamespace(
        encoding_for_model=lambda model: (_ for _ in ()).throw(KeyError(model)),
        get_encoding=lambda name: (_ for _ in ()).throw(KeyError(name)),
    )
    monkeypatch.setitem(sys.modules, "tiktoken", fake_module)

    result = estimate_token_count("12345678")
    assert result.count == 2
    assert result.method == "heuristic_char4"
    assert result.encoding is None


def test_estimated_chars_for_tokens() -> None:
    assert estimated_chars_for_tokens(0) == 0
    assert estimated_chars_for_tokens(10) == 40
