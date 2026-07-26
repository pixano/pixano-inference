# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API-key authentication for the inference server.

Authentication is optional: if no keys are configured (``PIXANO_INFERENCE_API_KEYS``),
the dependency is a no-op and a warning is emitted once at startup. When keys are set,
every guarded route requires a matching key supplied as either an ``X-API-Key`` header or
an ``Authorization: Bearer <key>`` header. Comparison is constant-time.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Header, HTTPException, status

from pixano_inference.server_settings import ServerSettings


logger = logging.getLogger(__name__)


def _extract_key(x_api_key: str | None, authorization: str | None) -> str | None:
    """Pull the presented key from the accepted headers."""
    if x_api_key:
        return x_api_key
    if authorization:
        scheme, _, credential = authorization.partition(" ")
        if scheme.lower() == "bearer" and credential:
            return credential
    return None


def _key_matches(presented: str, accepted: list[str]) -> bool:
    """Constant-time membership test that does not short-circuit on the first key."""
    matched = False
    for key in accepted:
        if secrets.compare_digest(presented, key):
            matched = True
    return matched


def make_api_key_dependency(settings: ServerSettings):
    """Build a FastAPI dependency enforcing API-key auth for the given settings.

    Args:
        settings: Server settings holding the accepted API keys.

    Returns:
        An async dependency callable. It is a no-op when no keys are configured.
    """
    accepted = list(settings.api_keys)

    async def require_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None),
    ) -> None:
        if not accepted:
            return
        presented = _extract_key(x_api_key, authorization)
        if presented is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Supply 'X-API-Key' or 'Authorization: Bearer <key>'.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not _key_matches(presented, accepted):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key.")

    return require_api_key


def warn_if_auth_disabled(settings: ServerSettings, host: str) -> None:
    """Emit a startup warning when auth is off, escalating on a non-loopback bind."""
    if settings.auth_enabled:
        return
    is_loopback = host in {"127.0.0.1", "localhost", "::1"}
    message = (
        "API-key authentication is DISABLED (no PIXANO_INFERENCE_API_KEYS configured). "
        "Every inference and admin endpoint is open to anyone who can reach the server."
    )
    if is_loopback:
        logger.warning(message)
    else:
        logger.error("%s The server is binding a non-loopback host (%s).", message, host)
