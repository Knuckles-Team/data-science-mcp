#!/usr/bin/python
"""Client accessors.

data-science-mcp has no external REST API of its own: all ML compute is delegated
to the Rust ``medulla`` engine via :class:`data_science_mcp.ml_engine.MLEngine` (a
process-local singleton; see ``MLEngine._rust_client``). Every served tool is
registered by hand in ``mcp_server.py`` and ``mcp/``; the engine's public methods
are not auto-derived into tools (a derived ``load_dataset`` bypassed the
data-root confinement of the hand-written tool). :func:`get_client` returns the
engine singleton for in-process callers.

:func:`get_rest_client` below is a separate, TLS-hardened accessor kept for an
optional upstream REST integration (``DATA_SCIENCE_MCP_URL`` /
``DATA_SCIENCE_MCP_TOKEN``); nothing in the live call path uses it today."""

import requests
from scion_sdk.config import setting
from scion_sdk.exceptions import AuthError, UnauthorizedError
from scion_sdk.tls.profile import ResolvedTLSProfile
from scion_sdk.tls.resolve import resolve_tls_profile

from data_science_mcp.ml_engine import MLEngine


def get_client() -> MLEngine:
    """Return the shared :class:`MLEngine` singleton (the tool surface's client)."""
    return MLEngine()


_rest_client = None


class _Client:
    """Small session holder for an optional upstream REST API seam."""

    def __init__(
        self,
        base_url: str,
        token: str,
        tls_profile: ResolvedTLSProfile,
    ) -> None:
        self.base_url = base_url
        self.tls_profile = tls_profile
        self.session = tls_profile.configure_requests_session(requests.Session())
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def close(self) -> None:
        """Release transport resources and runtime-only TLS material."""
        self.session.close()
        self.tls_profile.cleanup()


def get_rest_client(tls_profile: ResolvedTLSProfile | None = None):
    """Get or create a singleton TLS-hardened REST client (optional upstream seam).

    Not used by the MCP tool surface (see :func:`get_client`) — this is a
    ``DATA_SCIENCE_MCP_URL``/``DATA_SCIENCE_MCP_TOKEN``-configured client for an
    optional upstream REST integration.
    """
    global _rest_client
    if _rest_client is None:
        base_url = setting("DATA_SCIENCE_MCP_URL", "")
        token = setting("DATA_SCIENCE_MCP_TOKEN", "")
        if not base_url:
            raise RuntimeError("DATA_SCIENCE_MCP_URL is required")
        profile = tls_profile or resolve_tls_profile("data_science_mcp")

        try:
            if _rest_client is None:
                _rest_client = _Client(base_url, token, profile)
        except (AuthError, UnauthorizedError) as e:
            raise RuntimeError(
                "AUTHENTICATION ERROR: The configured credentials are not valid. "
                "Please check the configured credential and endpoint references."
            ) from e

    return _rest_client
