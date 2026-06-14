#!/usr/bin/python
"""Flat-token pretrain data prep (CONCEPT:ML-010).

Pure-numpy ``.npy`` path so it runs without h5py; exercises tokenize → flat array →
block reader round-trip and the streaming corpus shapes.
"""

from __future__ import annotations

import json

from data_science_mcp.data_engine import (
    load_token_array,
    prepare_pretrain_data,
    read_token_blocks,
    stream_corpus,
)


class ToyEnc:
    """Minimal tokenizer exposing ``.encode`` + ``eos_token_id`` (tiktoken-shaped)."""

    eos_token_id = 2

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % 50) + 3 for c in text]


def test_prepare_pretrain_data_npy_roundtrip(tmp_path):
    out = str(tmp_path / "toks.npy")
    tok = ToyEnc()
    rep = prepare_pretrain_data(
        [{"text": "abc"}, {"text": "de"}], tok, out, append_eos=True
    )
    assert rep["n_docs"] == 2
    # 3 + 1 EOS, 2 + 1 EOS = 7 tokens
    assert rep["n_tokens"] == 7
    assert rep["eos_id"] == 2
    arr = load_token_array(out)
    assert arr.tolist()[-1] == 2  # last token is the trailing EOS
    assert len(arr) == 7


def test_read_token_blocks_drops_remainder(tmp_path):
    out = str(tmp_path / "t.npy")
    tok = ToyEnc()
    prepare_pretrain_data(
        [{"text": "abcdef"}], tok, out, append_eos=False
    )  # 6 tokens
    blocks = list(read_token_blocks(out, block_size=4, drop_remainder=True))
    assert len(blocks) == 1 and len(blocks[0]) == 4
    blocks_keep = list(read_token_blocks(out, block_size=4, drop_remainder=False))
    assert len(blocks_keep) == 2 and len(blocks_keep[1]) == 2


def test_prepare_pretrain_data_limit(tmp_path):
    out = str(tmp_path / "l.npy")
    rep = prepare_pretrain_data(
        [{"text": "a"}, {"text": "b"}, {"text": "c"}], ToyEnc(), out, limit=2
    )
    assert rep["n_docs"] == 2


def test_stream_corpus_list_and_jsonl(tmp_path):
    # list of strings + dicts
    recs = list(stream_corpus(["hello", {"text": "world"}]))
    assert recs == [{"text": "hello"}, {"text": "world"}]
    # .jsonl file
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"text": "x"}) + "\n" + json.dumps({"text": "y"}) + "\n")
    assert [r["text"] for r in stream_corpus(str(p))] == ["x", "y"]


def test_prepare_pretrain_data_tool_plans_without_execute():
    from fastmcp import FastMCP

    from data_science_mcp.mcp.mcp_data_engine import register_data_engine_tools

    captured: dict = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn

            return deco

    register_data_engine_tools(_Recorder())
    assert "prepare_pretrain_data" in captured
    out = json.loads(
        captured["prepare_pretrain_data"](json.dumps(["a", "b"]), "out.h5", "{}")
    )
    assert out["executed"] is False and out["plan"]["format"] == "hdf5"
