"""MCP tools for the corpus curation engine (CONCEPT:DS-AHE.trainer.data-engine).

Exposes :mod:`data_science_mcp.data_engine` as action-routed MCP tools so an agent
(the ``data_curator`` persona) can build a clean training corpus: quality-filter →
deduplicate → decontaminate against eval sets → record dataset lineage. The heavy
near-duplicate search is offloaded to the epistemic-graph engine when available and
falls back to local cosine, so every tool works with or without a live engine.

All tools take/return JSON strings (parallel to :mod:`mcp_trainers`).
"""

import json
import re
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

_TOKENIZER_REPOSITORY_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?"
)
_CORPUS_SUFFIXES = (".jsonl", ".jsonl.zst", ".txt")
_TOKEN_OUTPUT_SUFFIXES = (".h5", ".hdf5", ".npy")
_MAX_CORPUS_SPEC_CHARS = 8 * 1024 * 1024
_MAX_OPTIONS_CHARS = 64 * 1024
_MAX_PRETRAIN_DOCS = 100_000
_MAX_DOC_CHARS = 1_000_000
_MAX_TOTAL_TOKENS = 50_000_000
_MAX_FLUSH_TOKENS = 1_000_000
_ALLOWED_TOKEN_DTYPES = frozenset({"int32", "int64", "uint16", "uint32"})


def register_data_engine_tools(mcp: FastMCP) -> None:
    """Register the corpus-curation tools (tag ``data-engine``)."""

    @mcp.tool(tags={"data-engine"})
    def dedup_corpus(records_json: str = "[]", options_json: str = "{}") -> str:
        """Remove exact + near-duplicate records (CONCEPT:DS-AHE.trainer.data-engine).

        Args:
            records_json: JSON list of ``{text: ...}`` records.
            options_json: ``{text_key, threshold, exact, near, use_engine}``.

        Returns:
            JSON ``{kept, removed:{exact,near}, n_in, n_out}``.
        """
        return _json(lambda: _dedup(records_json, options_json))

    @mcp.tool(tags={"data-engine"})
    def decontaminate_corpus(
        records_json: str = "[]", eval_texts_json: str = "[]", options_json: str = "{}"
    ) -> str:
        """Drop training records that leak held-out eval examples (CONCEPT:DS-AHE.trainer.data-engine).

        Args:
            records_json: JSON list of ``{text: ...}`` records.
            eval_texts_json: JSON list of eval text strings to protect against.
            options_json: ``{text_key, threshold, use_engine}``.
        """
        from data_science_mcp.data_engine import decontaminate  # noqa: PLC0415

        def _go() -> dict[str, Any]:
            recs = json.loads(records_json or "[]")
            ev = json.loads(eval_texts_json or "[]")
            opts = json.loads(options_json or "{}")
            return decontaminate(
                recs, ev, **_pick(opts, {"text_key", "threshold", "use_engine"})
            )

        return _json(_go)

    @mcp.tool(tags={"data-engine"})
    def curate_corpus(
        records_json: str = "[]", eval_texts_json: str = "[]", options_json: str = "{}"
    ) -> str:
        """Full curation pass: quality-filter → dedup → decontaminate → lineage.

        Args:
            records_json: JSON list of ``{text: ...}`` records.
            eval_texts_json: optional JSON list of eval texts to decontaminate against.
            options_json: merged options for each stage plus ``{name, version}`` for
                the dataset-lineage record.

        Returns:
            JSON ``{kept, stages:{quality,dedup,decontaminate}, provenance, n_in, n_out}``.
        """
        from data_science_mcp import data_engine as de  # noqa: PLC0415

        def _go() -> dict[str, Any]:
            recs = json.loads(records_json or "[]")
            ev = json.loads(eval_texts_json or "[]")
            opts = json.loads(options_json or "{}")
            n_in = len(recs)
            q = de.quality_filter(
                recs, **_pick(opts, {"text_key", "min_chars", "max_chars"})
            )
            d = de.dedup(
                q["kept"],
                **_pick(opts, {"text_key", "threshold", "exact", "near", "use_engine"}),
            )
            kept = d["kept"]
            dec = None
            if ev:
                dec = de.decontaminate(
                    kept, ev, **_pick(opts, {"text_key", "use_engine"})
                )
                kept = dec["kept"]
            ops = ["quality_filter", "dedup"] + (["decontaminate"] if ev else [])
            prov = de.dataset_provenance(
                opts.get("name", "corpus"),
                version=opts.get("version", "v1"),
                records=kept,
                parents=opts.get("parents"),
                ops=ops,
                kg_log=bool(opts.get("kg_log", False)),
            )
            return {
                "kept": kept,
                "stages": {
                    "quality": _counts(q),
                    "dedup": d.get("removed"),
                    "decontaminate": _counts(dec),
                },
                "provenance": prov,
                "n_in": n_in,
                "n_out": len(kept),
            }

        return _json(_go)

    @mcp.tool(tags={"data-engine"})
    def prepare_pretrain_data(
        corpus_spec_json: str, out_path: str, options_json: str = "{}"
    ) -> str:
        """Tokenize a corpus into a flat-token HDF5 file for pretraining (CONCEPT:DS-AHE.trainer.data-transformation).

        The large-scale data path for training from scratch: streams the corpus,
        encodes each doc (EOS-separated), and writes a contiguous ``tokens`` array
        that the pretrain trainer batches on the fly (no padding, bounded memory).

        Args:
            corpus_spec_json: JSON corpus spec — a list of ``{text}`` / strings, or a
                ``{"hf": "name", "split": ...}`` dict, or a path string to a
                ``.jsonl`` / ``.jsonl.zst`` / ``.txt`` file (see ``stream_corpus``).
            out_path: output path; ``.h5``/``.hdf5`` → HDF5, else a ``.npy`` array.
            options_json: ``{tokenizer (HF name or local dir, required to execute),
                text_key, append_eos, eos_id, limit, dtype, flush_every,
                execute: bool}``.

        Returns:
            JSON ``{plan, executed, out_path?, n_docs?, n_tokens?}`` (or ``{type(error).__name__}``).
        """

        def _go() -> dict[str, Any]:
            from data_science_mcp.data_engine import (  # noqa: PLC0415
                prepare_pretrain_data as _prep,
            )

            if len(corpus_spec_json) > _MAX_CORPUS_SPEC_CHARS:
                raise ValueError("corpus specification exceeds its size limit")
            if len(options_json) > _MAX_OPTIONS_CHARS:
                raise ValueError("pretraining options exceed their size limit")
            spec = json.loads(corpus_spec_json)
            opts = json.loads(options_json or "{}")
            if not isinstance(opts, dict):
                raise ValueError("pretraining options must be a JSON object")
            tok_ref = opts.get("tokenizer")
            plan = {
                "out_path": out_path,
                "format": "hdf5" if out_path.endswith((".h5", ".hdf5")) else "npy",
                "tokenizer": tok_ref,
                "revision": opts.get("revision"),
                "append_eos": opts.get("append_eos", True),
                "limit": opts.get("limit"),
            }
            if not opts.get("execute"):
                return {"plan": plan, "executed": False, "note": "set execute=true to run"}
            if not tok_ref:
                return {"plan": plan, "executed": False, "error": "options.tokenizer (HF name or local dir) is required to execute"}
            limit = opts.get("limit")
            flush_every = opts.get("flush_every", _MAX_FLUSH_TOKENS)
            dtype = opts.get("dtype", "int32")
            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or not 1 <= limit <= _MAX_PRETRAIN_DOCS
            ):
                return {
                    "plan": plan,
                    "executed": False,
                    "error": f"options.limit must be between 1 and {_MAX_PRETRAIN_DOCS}",
                }
            if (
                isinstance(flush_every, bool)
                or not isinstance(flush_every, int)
                or not 1 <= flush_every <= _MAX_FLUSH_TOKENS
                or dtype not in _ALLOWED_TOKEN_DTYPES
            ):
                return {
                    "plan": plan,
                    "executed": False,
                    "error": "invalid dtype or flush_every boundary",
                }
            from data_science_mcp.path_policy import data_root, resolve_data_path

            safe_out = resolve_data_path(out_path)
            if not safe_out.name.lower().endswith(_TOKEN_OUTPUT_SUFFIXES):
                return {
                    "plan": plan,
                    "executed": False,
                    "error": "out_path must end with .npy, .h5, or .hdf5",
                }
            safe_out.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(spec, str):
                safe_spec = resolve_data_path(spec, must_exist=True)
                if not safe_spec.name.lower().endswith(_CORPUS_SUFFIXES):
                    return {
                        "plan": plan,
                        "executed": False,
                        "error": "local corpus must be .txt, .jsonl, or .jsonl.zst",
                    }
                spec = str(safe_spec)
            elif isinstance(spec, dict) and "hf" in spec:
                dataset_id = spec.get("hf")
                if (
                    not isinstance(dataset_id, str)
                    or not _TOKENIZER_REPOSITORY_RE.fullmatch(dataset_id)
                    or any(key not in {"hf", "config", "split"} for key in spec)
                    or any(
                        value is not None
                        and (not isinstance(value, str) or len(value) > 256)
                        for key, value in spec.items()
                        if key in {"config", "split"}
                    )
                ):
                    return {
                        "plan": plan,
                        "executed": False,
                        "error": "invalid Hugging Face dataset reference",
                    }
            elif not isinstance(spec, list):
                return {
                    "plan": plan,
                    "executed": False,
                    "error": "corpus spec must be an inline list, dataset ID, or confined path",
                }

            if not isinstance(tok_ref, str):
                return {
                    "plan": plan,
                    "executed": False,
                    "error": "options.tokenizer must be a repository ID or local path",
                }
            token_path = Path(tok_ref).expanduser()
            local_tokenizer = (
                token_path.is_absolute()
                or tok_ref.startswith((".", "~"))
                or (data_root() / token_path).exists()
            )
            if local_tokenizer:
                tok_ref = str(resolve_data_path(tok_ref, must_exist=True))
            elif not _TOKENIZER_REPOSITORY_RE.fullmatch(tok_ref):
                return {
                    "plan": plan,
                    "executed": False,
                    "error": "invalid tokenizer repository ID",
                }
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415
            except ImportError:
                return {"plan": plan, "executed": False, "error": "transformers required — install data-science-mcp[training]"}
            from data_science_mcp.hf_security import require_pinned_revision  # noqa: PLC0415

            revision = require_pinned_revision(
                tok_ref,
                opts.get("revision"),
                local_files_only=local_tokenizer,
            )
            tok = AutoTokenizer.from_pretrained(
                tok_ref,
                revision=revision,
                trust_remote_code=False,
                local_files_only=local_tokenizer,
            )
            report = _prep(
                spec,
                tok,
                str(safe_out),
                max_doc_chars=_MAX_DOC_CHARS,
                max_tokens=_MAX_TOTAL_TOKENS,
                **_pick(
                    opts,
                    {"text_key", "append_eos", "eos_id", "limit", "dtype", "flush_every"},
                ),
            )
            report["out_path"] = out_path
            return {"plan": plan, "executed": True, **report}

        return _json(_go)

    @mcp.tool(tags={"data-engine"})
    def dataset_lineage(
        name: str, version: str = "v1", options_json: str = "{}"
    ) -> str:
        """Record a ``DatasetVersion`` provenance node (CONCEPT:DS-AHE.trainer.data-engine).

        Args:
            name: dataset name.
            version: dataset version tag.
            options_json: ``{n_records, parents, ops, extra, kg_log}``.
        """
        from data_science_mcp.data_engine import dataset_provenance  # noqa: PLC0415

        def _go() -> dict[str, Any]:
            opts = json.loads(options_json or "{}")
            return dataset_provenance(
                name,
                version=version,
                n_records=opts.get("n_records"),
                parents=opts.get("parents"),
                ops=opts.get("ops"),
                extra=opts.get("extra"),
                kg_log=bool(opts.get("kg_log", False)),
            )

        return _json(_go)


def _dedup(records_json: str, options_json: str) -> dict[str, Any]:
    from data_science_mcp.data_engine import dedup  # noqa: PLC0415

    recs = json.loads(records_json or "[]")
    opts = json.loads(options_json or "{}")
    return dedup(
        recs, **_pick(opts, {"text_key", "threshold", "exact", "near", "use_engine"})
    )


def _pick(opts: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: v for k, v in opts.items() if k in keys}


def _counts(stage: dict[str, Any] | None) -> dict[str, Any] | None:
    if stage is None:
        return None
    return {k: stage[k] for k in ("n_in", "n_out", "n_removed") if k in stage}


def _json(fn) -> str:
    try:
        return json.dumps(fn())
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid json: {type(e).__name__}"})
    except Exception:  # pragma: no cover - defensive
        return json.dumps({"error": "Operation failed"})


__all__ = ["register_data_engine_tools"]
