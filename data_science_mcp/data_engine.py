#!/usr/bin/python
"""Corpus streaming + curation engine (CONCEPT:DS-AHE.trainer.data-engine).

The data-quality front of the LLM-training stack: turn raw text into a clean,
deduplicated, decontaminated, packed corpus — because data quality, not optimizer
tricks, is what separates a good model from a bad one.

Everything here is **pure-Python correct** (so it is CPU-unit-testable with no heavy
deps) and uses two optional accelerators when present:

* **🤗 datasets** — memory-mapped/sharded streaming for large corpora
  (:func:`stream_corpus`); falls back to stdlib readers for ``.jsonl``/``.txt``/lists.
* **epistemic-graph** — the Rust HNSW/LSH ``find_similar_pairs`` kernel for fast
  near-duplicate and eval-set-leakage detection over the whole corpus
  (:func:`dedup`, :func:`decontaminate`); falls back to local cosine when the
  engine is unavailable.

Embeddings for similarity are computed locally with a deterministic hashing
vectorizer (:func:`feature_vector`) — no embedding model required — so dedup works
anywhere; the engine only accelerates the all-pairs search.

:func:`dataset_provenance` records the corpus as a ``Dataset``/``DatasetVersion``
node payload (lineage to parents) for the epistemic-graph, so every training run is
traceable back to the exact, curated data it consumed.

Concept: data-engine
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Callable, Iterable, Iterator

import numpy as np

logger = logging.getLogger("data_science_mcp.data_engine")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Streaming                                                                     #
# --------------------------------------------------------------------------- #
def stream_corpus(
    spec: Any, *, text_key: str = "text", hf_split: str = "train"
) -> Iterator[dict[str, Any]]:
    """Yield ``{text: ...}``-style records from many corpus shapes.

    ``spec`` may be: a list of dicts/strings; a path to a ``.jsonl``/``.txt`` file;
    or a ``{"hf": "dataset/name", "config": ..., "split": ...}`` dict routed through
    🤗 ``datasets.load_dataset(streaming=True)`` when the extra is installed.
    """
    if isinstance(spec, dict) and "hf" in spec:
        yield from _stream_hf(spec, hf_split)
        return
    if isinstance(spec, (list, tuple)):
        for r in spec:
            yield {text_key: r} if isinstance(r, str) else dict(r)
        return
    if isinstance(spec, str) and os.path.isfile(spec):
        is_jsonl = ".jsonl" in spec
        with _open_text(spec) as f:
            for line in f:
                line = line.strip() if is_jsonl else line.rstrip("\n")
                if not line.strip():
                    continue
                yield json.loads(line) if is_jsonl else {text_key: line}
        return
    raise ValueError(f"unsupported corpus spec: {type(spec).__name__}")


def _open_text(path: str) -> Any:
    """Open a corpus file for line iteration, transparently zstd-decompressing.

    Pile-style shards ship as ``.jsonl.zst``; this streams them through the
    ``zstandard`` decompressor (an optional dep, imported lazily) so the large
    pretraining corpora the from-scratch path consumes work without a manual
    decompress step.
    """
    if path.endswith(".zst"):
        try:
            import io  # noqa: PLC0415

            import zstandard  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - without the extra
            raise RuntimeError(
                "zstandard is required to stream .zst corpora; install "
                "`data-science-mcp[training]`"
            ) from e
        fh = open(path, "rb")  # noqa: SIM115 - wrapped reader closes it
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return open(path, encoding="utf-8")


def _stream_hf(spec: dict[str, Any], default_split: str) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - without the extra
        raise RuntimeError(
            "huggingface `datasets` is required for hf corpus specs; install "
            "`data-science-mcp[training]`"
        ) from e
    ds = load_dataset(
        spec["hf"],
        spec.get("config"),
        split=spec.get("split", default_split),
        streaming=True,
    )
    for row in ds:
        yield dict(row)


# --------------------------------------------------------------------------- #
# Similarity primitives (local; engine-accelerated all-pairs)                  #
# --------------------------------------------------------------------------- #
def _normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def _content_hash(text: str) -> str:
    return hashlib.sha1(_normalize_text(text).encode("utf-8")).hexdigest()


def feature_vector(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic hashing bag-of-tokens vector (L2-normalized) — no model needed."""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _TOKEN_RE.findall((text or "").lower()):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _near_pairs_local(
    vectors: np.ndarray, threshold: float
) -> list[tuple[int, int, float]]:
    """All-pairs cosine ≥ threshold (rows are pre-normalized) — O(n²) fallback."""
    if len(vectors) < 2:
        return []
    sims = vectors @ vectors.T
    pairs: list[tuple[int, int, float]] = []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            if s >= threshold:
                pairs.append((i, j, s))
    return pairs


def _near_pairs_engine(
    vectors: np.ndarray, ids: list[str], threshold: float
) -> list[tuple[int, int, float]] | None:
    """Try the epistemic-graph Rust ``find_similar_pairs`` (LSH/HNSW). ``None`` on miss.

    Reuses the cached :meth:`MLEngine._rust_client` singleton instead of opening a
    fresh connection (background thread + event loop + socket) per call — the old
    ``SyncEpistemicGraphClient.connect()`` here leaked one of each on every dedup.
    """
    try:  # pragma: no cover - requires a live engine + matching client
        from .ml_engine import MLEngine  # noqa: PLC0415 — lazy to avoid import cycle

        client = MLEngine._rust_client()
        if client is None:
            return None
        ds = getattr(client, "datascience", client)
        fn = getattr(ds, "find_similar_pairs", None)
        if fn is None:
            return None
        raw = fn(
            embeddings=[v.tolist() for v in vectors],
            ids=ids,
            threshold=threshold,
            use_lsh=len(vectors) > 4096,
        )
        idx = {sid: i for i, sid in enumerate(ids)}
        out: list[tuple[int, int, float]] = []
        for p in raw:
            a, b = p.get("a") or p.get("source"), p.get("b") or p.get("target")
            if a in idx and b in idx:
                out.append((idx[a], idx[b], float(p.get("similarity", threshold))))
        return out
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug("engine find_similar_pairs unavailable: %s", e)
        return None


#: Above this row count the local O(n²) near-duplicate fallback is refused rather
#: than silently burning CPU for minutes when the Rust engine is unavailable.
#: Override with DSM_NEAR_PAIRS_LOCAL_MAX (0 disables the cap).
_NEAR_PAIRS_LOCAL_MAX = int(os.environ.get("DSM_NEAR_PAIRS_LOCAL_MAX", "20000"))


def near_duplicate_pairs(
    texts: list[str], *, threshold: float = 0.9, dim: int = 256, use_engine: bool = True
) -> list[tuple[int, int, float]]:
    """Index pairs ``(i, j, sim)`` whose hashing-vector cosine ≥ ``threshold``."""
    vectors = (
        np.stack([feature_vector(t, dim) for t in texts])
        if texts
        else np.zeros((0, dim))
    )
    n = len(texts)
    if use_engine and n > 1:
        eng = _near_pairs_engine(vectors, [str(i) for i in range(n)], threshold)
        if eng is not None:
            return eng
        # Engine expected but unavailable → O(n²) local path. Make the perf cliff
        # LOUD (warning, not debug) and refuse above the cap, so a missing engine
        # surfaces instead of quietly running for minutes on a large corpus.
        if _NEAR_PAIRS_LOCAL_MAX and n > _NEAR_PAIRS_LOCAL_MAX:
            raise RuntimeError(
                f"near_duplicate_pairs: epistemic-graph engine unavailable and "
                f"n={n} exceeds the local O(n^2) cap ({_NEAR_PAIRS_LOCAL_MAX}). "
                f"Start the engine (EPISTEMIC_GRAPH_SOCKET / EPISTEMIC_GRAPH_TCP) "
                f"for the Rust LSH/HNSW path, raise DSM_NEAR_PAIRS_LOCAL_MAX, or "
                f"call with use_engine=False to force the local path."
            )
        logger.warning(
            "near_duplicate_pairs: engine unavailable; falling back to local "
            "O(n^2) cosine for n=%d (set EPISTEMIC_GRAPH_SOCKET/TCP for the Rust "
            "LSH/HNSW path)",
            n,
        )
    return _near_pairs_local(vectors, threshold)


# --------------------------------------------------------------------------- #
# Curation ops                                                                  #
# --------------------------------------------------------------------------- #
def dedup(
    records: Iterable[dict[str, Any]],
    *,
    text_key: str = "text",
    threshold: float = 0.95,
    exact: bool = True,
    near: bool = True,
    use_engine: bool = True,
) -> dict[str, Any]:
    """Remove exact and near-duplicate records (keeps the first of each cluster).

    Returns ``{kept, removed, n_in, n_out}`` where ``kept`` is the deduplicated list.
    """
    recs = [dict(r) for r in records]
    keep = [True] * len(recs)
    removed_exact = 0
    if exact:
        seen: set[str] = set()
        for i, r in enumerate(recs):
            h = _content_hash(str(r.get(text_key, "")))
            if h in seen:
                keep[i] = False
                removed_exact += 1
            else:
                seen.add(h)
    removed_near = 0
    if near:
        live = [i for i in range(len(recs)) if keep[i]]
        texts = [str(recs[i].get(text_key, "")) for i in live]
        for li, lj, _sim in near_duplicate_pairs(
            texts, threshold=threshold, use_engine=use_engine
        ):
            gi, gj = live[li], live[lj]
            if keep[gi] and keep[gj]:
                keep[gj] = False  # drop the later record of the pair
                removed_near += 1
    kept = [recs[i] for i in range(len(recs)) if keep[i]]
    return {
        "kept": kept,
        "removed": {"exact": removed_exact, "near": removed_near},
        "n_in": len(recs),
        "n_out": len(kept),
    }


def decontaminate(
    records: Iterable[dict[str, Any]],
    eval_texts: list[str],
    *,
    text_key: str = "text",
    threshold: float = 0.9,
    use_engine: bool = True,
) -> dict[str, Any]:
    """Drop training records too similar to any held-out eval text (leakage guard)."""
    recs = [dict(r) for r in records]
    if not eval_texts or not recs:
        return {"kept": recs, "n_removed": 0, "n_in": len(recs), "n_out": len(recs)}
    eval_vecs = np.stack([feature_vector(t) for t in eval_texts])
    kept: list[dict[str, Any]] = []
    removed = 0
    for r in recs:
        v = feature_vector(str(r.get(text_key, "")))
        if float(np.max(eval_vecs @ v)) >= threshold:
            removed += 1
        else:
            kept.append(r)
    return {"kept": kept, "n_removed": removed, "n_in": len(recs), "n_out": len(kept)}


def quality_filter(
    records: Iterable[dict[str, Any]],
    *,
    text_key: str = "text",
    min_chars: int = 1,
    max_chars: int | None = None,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Length + custom-predicate quality gate."""
    recs = [dict(r) for r in records]
    kept: list[dict[str, Any]] = []
    for r in recs:
        text = str(r.get(text_key, ""))
        n = len(text.strip())
        if n < min_chars:
            continue
        if max_chars is not None and n > max_chars:
            continue
        if predicate is not None and not predicate(r):
            continue
        kept.append(r)
    return {"kept": kept, "n_in": len(recs), "n_out": len(kept)}


def pack_sequences(
    token_id_lists: list[list[int]],
    max_len: int,
    *,
    eos_id: int | None = None,
    drop_remainder: bool = True,
) -> list[list[int]]:
    """Concatenate (with optional EOS separators) and split into ``max_len`` blocks.

    The standard pretraining throughput trick: no padding waste — every block is a
    full ``max_len`` window of real tokens.
    """
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    flat: list[int] = []
    for ids in token_id_lists:
        flat.extend(ids)
        if eos_id is not None:
            flat.append(eos_id)
    blocks = [flat[i : i + max_len] for i in range(0, len(flat), max_len)]
    if drop_remainder and blocks and len(blocks[-1]) < max_len:
        blocks.pop()
    return blocks


# --------------------------------------------------------------------------- #
# Large-scale pretrain data prep (flat-token HDF5)  (CONCEPT:DS-AHE.trainer.data-transformation)            #
# --------------------------------------------------------------------------- #
def _token_encode_fn(tokenizer: Any) -> Callable[[str], list[int]]:
    """Adapt any tokenizer to ``text -> list[int]`` via its ``.encode``.

    Works for an HF ``AutoTokenizer``, a ``tiktoken`` ``Encoding``, our trained BPE
    tokenizer, or a toy object in tests — all expose ``.encode(text)``.
    """
    enc = getattr(tokenizer, "encode", None)
    if not callable(enc):
        raise ValueError("tokenizer must expose a callable .encode(text) -> list[int]")
    return lambda t: list(enc(t))


def prepare_pretrain_data(
    spec: Any,
    tokenizer: Any,
    out_path: str,
    *,
    text_key: str = "text",
    append_eos: bool = True,
    eos_id: int | None = None,
    limit: int | None = None,
    dtype: str = "int32",
    flush_every: int = 1_000_000,
) -> dict[str, Any]:
    """Tokenize a streamed corpus into a flat 1-D token array on disk (CONCEPT:DS-AHE.trainer.data-transformation).

    The pretraining-throughput data path the from-scratch trainer needs: stream
    ``spec`` (list / ``.jsonl`` / ``.jsonl.zst`` / HF dataset — see
    :func:`stream_corpus`), encode each doc with ``tokenizer.encode`` (appending the
    EOS so doc boundaries survive), and append to a contiguous ``tokens`` dataset.
    Per-example boundaries are recovered on the fly at batch time
    (:func:`read_token_blocks`), so there is no padding and no per-example pickling.

    ``out_path`` ending ``.h5``/``.hdf5`` writes a resizable HDF5 ``tokens`` dataset
    (streamed in ``flush_every``-token chunks, so memory stays bounded on huge
    corpora); any other extension writes a single ``.npy`` array (dependency-light,
    used by the CPU tests). Returns ``{out_path, n_docs, n_tokens, dtype, eos_id}``.
    """
    encode = _token_encode_fn(tokenizer)
    if append_eos and eos_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
    is_h5 = out_path.endswith((".h5", ".hdf5"))
    buf: list[np.ndarray] = []
    npy_chunks: list[np.ndarray] = []
    n_docs = n_tokens = 0
    h5f = dset = None
    try:
        if is_h5:
            try:
                import h5py  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover - without the extra
                raise RuntimeError(
                    "h5py is required for HDF5 token output; install "
                    "`data-science-mcp[training]` or use a .npy out_path"
                ) from e
            h5f = h5py.File(out_path, "w")
            dset = h5f.create_dataset(
                "tokens", (0,), maxshape=(None,), dtype=dtype, chunks=True
            )

        def _flush() -> None:
            nonlocal buf
            if not buf:
                return
            arr = np.concatenate(buf) if len(buf) > 1 else buf[0]
            if is_h5:
                old = dset.shape[0]
                dset.resize((old + arr.size,))
                dset[old:] = arr
            else:
                npy_chunks.append(arr)
            buf = []

        for rec in stream_corpus(spec, text_key=text_key):
            ids = encode(str(rec.get(text_key, "")))
            if append_eos and eos_id is not None:
                ids.append(int(eos_id))
            if not ids:
                continue
            buf.append(np.asarray(ids, dtype=dtype))
            n_docs += 1
            n_tokens += len(ids)
            if sum(b.size for b in buf) >= flush_every:
                _flush()
            if limit is not None and n_docs >= limit:
                break
        _flush()
        if is_h5:
            h5f.attrs["n_docs"] = n_docs
            h5f.attrs["n_tokens"] = n_tokens
            h5f.attrs["eos_id"] = -1 if eos_id is None else int(eos_id)
        else:
            arr = (
                np.concatenate(npy_chunks)
                if npy_chunks
                else np.zeros((0,), dtype=dtype)
            )
            np.save(out_path, arr)
    finally:
        if h5f is not None:
            h5f.close()
    return {
        "out_path": out_path,
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "dtype": dtype,
        "eos_id": eos_id,
    }


def load_token_array(path: str) -> np.ndarray:
    """Load the flat token array written by :func:`prepare_pretrain_data`."""
    if path.endswith((".h5", ".hdf5")):
        import h5py  # noqa: PLC0415

        with h5py.File(path, "r") as f:
            return f["tokens"][:]
    if not path.endswith(".npy"):
        path = path + ".npy"
    return np.load(path)


def read_token_blocks(
    path: str, block_size: int, *, drop_remainder: bool = True
) -> Iterator[list[int]]:
    """Yield contiguous ``block_size`` token windows from a prepared token file.

    The on-the-fly batch-boundary read: a flat token stream is sliced into fixed
    windows at train time (no stored per-example structure). The trailing partial
    window is dropped unless ``drop_remainder=False``.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    tokens = load_token_array(path)
    n = len(tokens)
    end = (n // block_size) * block_size if drop_remainder else n
    for i in range(0, end, block_size):
        yield tokens[i : i + block_size].tolist()


# --------------------------------------------------------------------------- #
# Provenance                                                                    #
# --------------------------------------------------------------------------- #
def dataset_provenance(
    name: str,
    *,
    version: str,
    records: list[dict[str, Any]] | None = None,
    n_records: int | None = None,
    parents: list[str] | None = None,
    ops: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    kg_log: bool = False,
) -> dict[str, Any]:
    """Build (and optionally KG-persist) a ``DatasetVersion`` lineage payload."""
    count = n_records if n_records is not None else (len(records) if records else 0)
    fp = ""
    if records is not None:
        h = hashlib.sha1()
        for r in records:
            h.update(_content_hash(str(r.get("text", r))).encode("utf-8"))
        fp = h.hexdigest()
    payload = {
        "kind": "DatasetVersion",
        "name": name,
        "version": version,
        "n_records": count,
        "fingerprint": fp,
        "parents": list(parents or []),
        "ops": list(ops or []),
        "extra": dict(extra or {}),
    }
    if kg_log:
        _kg_write(payload)
    return payload


def _kg_write(payload: dict[str, Any]) -> None:
    try:  # pragma: no cover - requires a live KG facade/engine
        from agent_utilities.knowledge_graph.facade import (  # noqa: PLC0415
            KnowledgeGraphFacade,
        )

        kg = KnowledgeGraphFacade()
        writer = getattr(kg, "write", None) or getattr(kg, "add_node", None)
        if writer is not None:
            writer(payload)
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug("KG dataset provenance skipped: %s", e)


__all__ = [
    "stream_corpus",
    "feature_vector",
    "near_duplicate_pairs",
    "dedup",
    "decontaminate",
    "quality_filter",
    "pack_sequences",
    "prepare_pretrain_data",
    "load_token_array",
    "read_token_blocks",
    "dataset_provenance",
]
