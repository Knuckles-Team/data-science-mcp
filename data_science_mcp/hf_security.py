"""Fail-closed Hugging Face artifact revision policy."""

from __future__ import annotations

import re
from pathlib import Path

_COMMIT_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")


def require_pinned_revision(
    reference: object,
    revision: object = None,
    *,
    local_files_only: bool = False,
) -> str | None:
    """Require an immutable commit hash for every remote Hub artifact.

    Existing local paths do not need a Hub revision. Branches and tags are
    mutable and therefore intentionally rejected for remote repositories.
    """
    rendered_reference = str(reference or "").strip()
    path = Path(rendered_reference).expanduser()
    is_local = local_files_only or path.exists()
    if is_local:
        return str(revision).strip() if revision else None
    rendered_revision = str(revision or "").strip()
    if not _COMMIT_REVISION.fullmatch(rendered_revision):
        raise ValueError(
            "remote Hugging Face artifacts require a full immutable commit revision"
        )
    return rendered_revision.lower()


__all__ = ["require_pinned_revision"]
