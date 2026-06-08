"""Shared test fixtures for Data Science Mcp.

Compute runs in the Rust epistemic-graph engine, so the engine-backed tests need
a live server. The session-scoped `epistemic_graph_engine` fixture best-effort
launches one (installed `epistemic-graph-server` console script, or a sibling
source build) and points `EPISTEMIC_GRAPH_SOCKET` at it. The per-test
`require_engine` fixture depends on it and skips only if no engine could be
started/reached — so coverage runs whenever the binary is available, and degrades
to a clean skip when it is not.
"""

import importlib.util
import os
import shutil
import subprocess
import tempfile
import time

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("DATA_URL", "https://test.example.com")
    monkeypatch.setenv("DATA_TOKEN", "test-token-12345")
    monkeypatch.setenv("DATA_SSL_VERIFY", "False")


def _find_engine_binary() -> str | None:
    """Locate the epistemic-graph-server binary: prefer the installed console
    script (epistemic-graph[datascience] ships the version-matched binary via
    maturin bindings=bin), then fall back to the NEWEST sibling source build
    (newest mtime avoids picking a stale release over a fresh debug build)."""
    found = shutil.which("epistemic-graph-server")
    if found:
        return found
    here = os.path.dirname(__file__)
    candidates = [
        os.path.abspath(
            os.path.join(
                here,
                "..",
                "..",
                "..",
                "epistemic-graph",
                "target",
                build,
                "epistemic-graph-server",
            )
        )
        for build in ("release", "debug")
    ]
    existing = [c for c in candidates if os.path.isfile(c)]
    if not existing:
        return None
    return max(existing, key=os.path.getmtime)


@pytest.fixture(scope="session")
def epistemic_graph_engine():
    """Launch a local epistemic-graph engine for the session (best effort).

    No-ops (yields None) when an external engine is already configured, or when
    no binary is available — in which case `require_engine` tests skip.
    """
    # Respect an externally provided engine (e.g. a CI service container).
    if os.environ.get("EPISTEMIC_GRAPH_SOCKET") or os.environ.get(
        "EPISTEMIC_GRAPH_TCP"
    ):
        yield None
        return

    binary = _find_engine_binary()
    if binary is None:
        yield None
        return

    tmpdir = tempfile.mkdtemp(prefix="ds-mcp-eg-")
    sock = os.path.join(tmpdir, "engine.sock")
    proc = subprocess.Popen(
        [binary, "--socket-path", sock],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for _ in range(60):
        if os.path.exists(sock) or proc.poll() is not None:
            break
        time.sleep(0.5)

    if os.path.exists(sock):
        os.environ["EPISTEMIC_GRAPH_SOCKET"] = sock

    try:
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        if os.path.exists(sock):
            try:
                os.remove(sock)
            except OSError:
                pass
        os.environ.pop("EPISTEMIC_GRAPH_SOCKET", None)


@pytest.fixture
def require_engine(epistemic_graph_engine):
    """Skip a test unless the epistemic-graph compute engine is reachable.

    Depends on the session engine fixture (which starts one when possible), then
    re-probes with the current env. Compute (fit/predict/evaluate/cross-validate)
    runs entirely in the engine.
    """
    from data_science_mcp.ml_engine import MLEngine, _UNPROBED

    MLEngine._rust_client_cache = _UNPROBED  # re-probe with current env
    if MLEngine._rust_client() is None:
        pytest.skip("epistemic-graph engine not reachable")


@pytest.fixture
def require_sklearn():
    """Skip a test unless scikit-learn is installed (optional `[datasets]` extra,
    used only by the built-in sample-dataset loaders)."""
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn not installed (install data-science-mcp[datasets])")
