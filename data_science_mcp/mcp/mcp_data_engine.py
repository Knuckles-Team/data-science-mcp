"""MCP tools for the corpus curation engine (CONCEPT:DS-AHE.trainer.data-engine).

Exposes :mod:`data_science_mcp.data_engine` as action-routed MCP tools so an agent
(the ``data_curator`` persona) can build a clean training corpus: quality-filter →
deduplicate → decontaminate against eval sets → record dataset lineage. The heavy
near-duplicate search is offloaded to the epistemic-graph engine when available and
falls back to local cosine, so every tool works with or without a live engine.

All tools take/return JSON strings (parallel to :mod:`mcp_trainers`).
"""

import json
from typing import Any

from fastmcp import FastMCP


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
            JSON ``{plan, executed, out_path?, n_docs?, n_tokens?}`` (or ``{error}``).
        """

        def _go() -> dict[str, Any]:
            from data_science_mcp.data_engine import (  # noqa: PLC0415
                prepare_pretrain_data as _prep,
            )

            spec = json.loads(corpus_spec_json)
            opts = json.loads(options_json or "{}")
            tok_ref = opts.get("tokenizer")
            plan = {
                "out_path": out_path,
                "format": "hdf5" if out_path.endswith((".h5", ".hdf5")) else "npy",
                "tokenizer": tok_ref,
                "append_eos": opts.get("append_eos", True),
                "limit": opts.get("limit"),
            }
            if not opts.get("execute"):
                return {"plan": plan, "executed": False, "note": "set execute=true to run"}
            if not tok_ref:
                return {"plan": plan, "executed": False, "error": "options.tokenizer (HF name or local dir) is required to execute"}
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415
            except ImportError:
                return {"plan": plan, "executed": False, "error": "transformers required — install data-science-mcp[training]"}
            tok = AutoTokenizer.from_pretrained(tok_ref)
            report = _prep(
                spec,
                tok,
                out_path,
                **_pick(
                    opts,
                    {"text_key", "append_eos", "eos_id", "limit", "dtype", "flush_every"},
                ),
            )
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
        return json.dumps({"error": f"invalid json: {e}"})
    except Exception as e:  # pragma: no cover - defensive
        return json.dumps({"error": str(e)})


__all__ = ["register_data_engine_tools"]
