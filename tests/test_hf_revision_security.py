from __future__ import annotations

import pytest

from data_science_mcp.hf_security import require_pinned_revision


def test_remote_hub_artifacts_require_immutable_commit_revision() -> None:
    with pytest.raises(ValueError, match="immutable commit"):
        require_pinned_revision("org/model")
    with pytest.raises(ValueError, match="immutable commit"):
        require_pinned_revision("org/model", "main")

    revision = "a" * 40
    assert require_pinned_revision("org/model", revision) == revision


def test_local_hub_artifacts_do_not_require_remote_revision(tmp_path) -> None:
    local_model = tmp_path / "model"
    local_model.mkdir()
    assert require_pinned_revision(local_model) is None
    assert require_pinned_revision(
        "virtual-local-model", local_files_only=True
    ) is None
