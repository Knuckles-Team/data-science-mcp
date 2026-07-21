"""Resource-boundary tests for MCP-triggered pretraining preparation."""

import pytest

from data_science_mcp.data_engine import prepare_pretrain_data


class _Tokenizer:
    eos_token_id = None

    @staticmethod
    def encode(text):
        return list(range(len(text)))


def test_pretrain_rejects_oversized_document_before_write(tmp_path):
    output = tmp_path / "tokens.npy"

    with pytest.raises(ValueError, match="document exceeds"):
        prepare_pretrain_data(
            ["too large"],
            _Tokenizer(),
            str(output),
            max_doc_chars=4,
        )


def test_pretrain_rejects_total_token_overflow(tmp_path):
    output = tmp_path / "tokens.npy"

    with pytest.raises(ValueError, match="token output exceeds"):
        prepare_pretrain_data(
            ["abc", "def"],
            _Tokenizer(),
            str(output),
            max_tokens=5,
        )
