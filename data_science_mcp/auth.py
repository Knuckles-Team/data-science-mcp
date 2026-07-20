#!/usr/bin/python
# coding: utf-8

import requests

from agent_utilities.core.config import setting
from agent_utilities.core.exceptions import AuthError, UnauthorizedError
from agent_utilities.core.transport_security import (
    ResolvedTLSProfile,
    resolve_configured_tls_profile,
)

_client = None


class _Client:
    """Small session holder for the package's upstream API seam."""

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


def get_client(tls_profile: ResolvedTLSProfile | None = None):
    """Get or create a singleton API client instance."""
    global _client
    if _client is None:
        base_url = setting("DATA_SCIENCE_MCP_URL", "")
        token = setting("DATA_SCIENCE_MCP_TOKEN", "")
        if not base_url:
            raise RuntimeError("DATA_SCIENCE_MCP_URL is required")
        profile = tls_profile or resolve_configured_tls_profile("data_science_mcp")

        try:
            if _client is None:
                _client = _Client(base_url, token, profile)
        except (AuthError, UnauthorizedError) as e:
            raise RuntimeError(
                "AUTHENTICATION ERROR: The configured credentials are not valid. "
                "Please check the configured credential and endpoint references."
            ) from e

    return _client
