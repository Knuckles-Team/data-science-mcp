#!/usr/bin/python
"""Tests for the corpus curation engine (CONCEPT:DS-AHE.trainer.data-engine).

Pure-Python coverage (``use_engine=False`` for determinism); the epistemic-graph
all-pairs acceleration is exercised opportunistically only when a live engine is
present and is behaviourally equivalent to the local fallback tested here.
"""

from __future__ import annotations

import hashlib
import json

from data_science_mcp import data_engine as de


def test_dedup_exact_and_near():
    records = [
        {"text": "the quick brown fox jumps"},
        {"text": "The Quick brown Fox jumps"},  # exact after normalization
        {"text": "the quick brown fox jumps over"},  # near-duplicate
        {"text": "completely unrelated sentence about ships"},
    ]
    out = de.dedup(records, threshold=0.9, use_engine=False)
    assert out["removed"]["exact"] == 1
    assert out["removed"]["near"] >= 1
    assert out["n_out"] <= 2
    # The unrelated sentence must survive.
    assert any("ships" in r["text"] for r in out["kept"])


def test_dedup_keeps_first_of_cluster():
    records = [{"text": "alpha beta gamma"}, {"text": "alpha beta gamma"}]
    out = de.dedup(records, use_engine=False)
    assert out["n_out"] == 1
    assert out["kept"][0]["text"] == "alpha beta gamma"


def test_decontaminate_drops_leaked():
    records = [
        {"text": "what is the capital of france paris"},
        {"text": "a wholly different training example here"},
    ]
    eval_texts = ["what is the capital of france paris"]
    out = de.decontaminate(records, eval_texts, threshold=0.85, use_engine=False)
    assert out["n_removed"] == 1
    assert out["n_out"] == 1
    assert "different" in out["kept"][0]["text"]


def test_quality_filter_length_and_predicate():
    records = [
        {"text": "ok length text"},
        {"text": "x"},  # too short
        {"text": "drop me", "lang": "fr"},
    ]
    out = de.quality_filter(
        records, min_chars=3, predicate=lambda r: r.get("lang", "en") == "en"
    )
    assert out["n_out"] == 1
    assert out["kept"][0]["text"] == "ok length text"


def test_pack_sequences_blocks_and_remainder():
    blocks = de.pack_sequences([[1, 2, 3], [4, 5], [6, 7, 8, 9]], max_len=4)
    # 9 tokens → two full blocks of 4, remainder of 1 dropped.
    assert blocks == [[1, 2, 3, 4], [5, 6, 7, 8]]
    kept = de.pack_sequences([[1, 2, 3, 4, 5]], max_len=4, drop_remainder=False)
    assert kept == [[1, 2, 3, 4], [5]]
    with_eos = de.pack_sequences([[1, 2], [3, 4]], max_len=3, eos_id=0)
    assert with_eos[0] == [1, 2, 0]


def test_dataset_provenance_payload_and_fingerprint_stable():
    recs = [{"text": "a b c"}, {"text": "d e f"}]
    p1 = de.dataset_provenance("corpus", version="v1", records=recs, ops=["dedup"])
    p2 = de.dataset_provenance("corpus", version="v1", records=recs, ops=["dedup"])
    assert p1["kind"] == "DatasetVersion"
    assert p1["n_records"] == 2
    assert p1["ops"] == ["dedup"]
    assert p1["fingerprint"] and p1["fingerprint"] == p2["fingerprint"]
    assert len(p1["fingerprint"]) == hashlib.sha256().digest_size * 2
    expected = hashlib.sha256()
    for record in recs:
        expected.update(hashlib.sha256(record["text"].encode()).hexdigest().encode())
    assert p1["fingerprint"] == expected.hexdigest()


def test_content_and_token_hashing_use_sha256_width():
    normalized = b"alpha beta"
    assert de._content_hash("Alpha beta") == hashlib.sha256(normalized).hexdigest()

    dimension = 1024
    vector = de.feature_vector("alpha", dim=dimension)
    expected_index = int.from_bytes(hashlib.sha256(b"alpha").digest(), "big") % dimension
    assert vector[expected_index] == 1.0
    assert int((vector != 0).sum()) == 1


def test_stream_corpus_from_list_and_jsonl(tmp_path):
    got = list(de.stream_corpus(["hello", "world"]))
    assert got == [{"text": "hello"}, {"text": "world"}]
    jf = tmp_path / "c.jsonl"
    jf.write_text('{"text": "one"}\n{"text": "two"}\n', encoding="utf-8")
    rows = list(de.stream_corpus(str(jf)))
    assert rows == [{"text": "one"}, {"text": "two"}]


# --------------------------------------------------------------------------- #
# MCP tool surface                                                              #
# --------------------------------------------------------------------------- #
def test_data_engine_tools_register_and_curate():
    from data_science_mcp.mcp.mcp_data_engine import register_data_engine_tools

    captured: dict = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn

            return deco

    register_data_engine_tools(_Recorder())
    assert {
        "dedup_corpus",
        "decontaminate_corpus",
        "curate_corpus",
        "dataset_lineage",
    } <= set(captured)

    out = json.loads(
        captured["curate_corpus"](
            json.dumps(
                [
                    {"text": "alpha beta gamma delta"},
                    {"text": "alpha beta gamma delta"},  # dup
                    {"text": "x"},  # too short
                    {"text": "unique content that should survive curation"},
                ]
            ),
            json.dumps(["alpha beta gamma delta"]),  # eval text → decontaminate
            json.dumps(
                {
                    "min_chars": 3,
                    "threshold": 0.9,
                    "use_engine": False,
                    "name": "t",
                    "version": "v1",
                }
            ),
        )
    )
    assert out["provenance"]["kind"] == "DatasetVersion"
    assert out["n_out"] >= 1
    assert any("survive" in r["text"] for r in out["kept"])


def test_near_duplicate_pairs_loud_bounded_fallback(monkeypatch):
    """B5: engine-miss fallback is loud + capped, never a silent O(n^2) cliff."""
    import pytest

    # Force the engine to "miss" so the local fallback path is taken.
    monkeypatch.setattr(de, "_near_pairs_engine", lambda *a, **k: None)
    monkeypatch.setattr(de, "_NEAR_PAIRS_LOCAL_MAX", 5)

    # Over the cap with use_engine=True (engine expected) → refuse loudly.
    with pytest.raises(RuntimeError):
        de.near_duplicate_pairs([f"t{i}" for i in range(6)])

    # Below the cap → warn + fall back to local, returning a list.
    out = de.near_duplicate_pairs([f"doc number {i}" for i in range(3)])
    assert isinstance(out, list)

    # Explicit opt-out bypasses both the engine and the cap.
    out2 = de.near_duplicate_pairs([f"t{i}" for i in range(6)], use_engine=False)
    assert isinstance(out2, list)
