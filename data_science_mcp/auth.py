#!/usr/bin/python
# coding: utf-8

import os
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from agent_utilities.core.exceptions import AuthError, UnauthorizedError

# TODO: Import your API wrapper class here
_client = None


def get_client():
    """Get or create a singleton API client instance."""
    global _client
    if _client is None:
        base_url = os.getenv("DATA_SCIENCE_MCP_URL", "http://localhost:8080")
        token = os.getenv("DATA_SCIENCE_MCP_TOKEN", "")
        verify = os.getenv("DATA_SCIENCE_MCP_SSL_VERIFY") or os.getenv("DATA_SCIENCE_MCP_VERIFY", "True").lower() in (
            "true",
            "1",
            "yes",
        )

        try:
            # TODO: Uncomment and configure once the API wrapper class is created
            # _client = DataScienceApiClient(
            #     base_url=base_url,
            #     token=token,
            #     verify=verify,
            # )

            # Placeholder until API wrapper is implemented
            if _client is None:
                session = requests.Session()
                session.headers.update({"Authorization": f"Bearer {token}"})
                session.verify = verify
                _client = type(
                    "Client", (), {"session": session, "base_url": base_url}
                )()
        except (AuthError, UnauthorizedError) as e:
            raise RuntimeError(
                f"AUTHENTICATION ERROR: The credentials provided are not valid for '{base_url}'. "
                f"Please check your DATA_SCIENCE_MCP_TOKEN and DATA_SCIENCE_MCP_URL environment variables. "
                f"Error details: {str(e)}"
            ) from e

    return _client
