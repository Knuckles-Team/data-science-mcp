"""Security tests for MCP-triggered data file paths."""

from pathlib import Path

import pytest

from data_science_mcp.path_policy import (
    authorized_dataset_source,
    resolve_data_path,
)


def test_relative_data_path_is_confined(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_SCIENCE_DATA_ROOT", str(tmp_path))
    source = tmp_path / "inputs" / "sample.csv"
    source.parent.mkdir()
    source.write_text("x,y\n1,2\n")

    assert resolve_data_path("inputs/sample.csv", must_exist=True) == source
    assert authorized_dataset_source("inputs/sample.csv") == str(source)
    assert authorized_dataset_source("iris") is None


@pytest.mark.parametrize("value", ["../outside.csv", "../../outside.csv"])
def test_data_path_rejects_traversal(monkeypatch, tmp_path, value):
    monkeypatch.setenv("DATA_SCIENCE_DATA_ROOT", str(tmp_path / "data"))

    with pytest.raises(ValueError, match="escapes"):
        resolve_data_path(value)


def test_data_path_rejects_symlink_escape(monkeypatch, tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("x,y\n1,2\n")
    link = root / "linked.csv"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are unavailable")
    monkeypatch.setenv("DATA_SCIENCE_DATA_ROOT", str(root))

    with pytest.raises(ValueError, match="escapes|symbolic links"):
        resolve_data_path(str(Path("linked.csv")), must_exist=True)


def test_data_path_rejects_symlink_even_when_target_stays_inside_root(
    monkeypatch, tmp_path
):
    root = tmp_path / "data"
    root.mkdir()
    target = root / "target.csv"
    target.write_text("x,y\n1,2\n")
    link = root / "linked.csv"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable")
    monkeypatch.setenv("DATA_SCIENCE_DATA_ROOT", str(root))

    with pytest.raises(ValueError, match="symbolic links"):
        resolve_data_path("linked.csv", must_exist=True)
