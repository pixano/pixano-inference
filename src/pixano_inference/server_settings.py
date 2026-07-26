# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Server-side runtime settings (security, media policy, logging).

These are read from environment variables prefixed ``PIXANO_INFERENCE_`` (or a ``.env``
file) and drive the request-facing behaviour of the server: API-key authentication, the
media-ingestion security policy, CORS, and request-body limits.

``ServerSettings`` is what the ingress and model replicas consult at request time; because
it derives entirely from the environment, a Ray worker process reconstructs the same policy
its driver used (see :mod:`pixano_inference.utils.media`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


_MB = 1024 * 1024


def _split_csv(value: Any) -> Any:
    """Allow list-valued settings to be given as a comma-separated string in env vars."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        # Fall back to comma-splitting for the common "a,b,c" env form. JSON arrays are
        # still accepted by pydantic-settings when the string looks like a list.
        if stripped.startswith("["):
            return value
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return value


class ServerSettings(BaseSettings):
    """Request-facing server configuration, sourced from the environment.

    Attributes:
        api_keys: Accepted API keys. When empty, authentication is disabled (a warning is
            emitted at startup; see :mod:`pixano_inference.security`).
        media_allow_url: Whether URL media (http/https) may be fetched at all.
        media_url_host_allowlist: Hostnames whose private/loopback IPs are permitted
            despite the SSRF guard (e.g. ``pixano`` on a Docker network).
        media_allow_private_ips: Disable the private/loopback/link-local IP block entirely
            (unsafe; for trusted single-host setups only).
        media_roots: Filesystem roots under which local-path media is allowed. Empty means
            local-path media is denied outright.
        media_connect_timeout_s: Connect timeout for URL fetches.
        media_read_timeout_s: Read timeout for URL fetches.
        media_max_redirects: Maximum number of HTTP redirects followed (each re-validated).
        media_max_image_bytes: Maximum decoded size for a fetched image.
        media_max_video_bytes: Maximum size for a fetched video.
        cors_allow_origins: Allowed CORS origins. Empty disables CORS.
        max_request_body_bytes: Maximum accepted request body size.
        log_level: Root log level applied at startup.
        log_json: Emit structured JSON logs when true.
    """

    model_config = SettingsConfigDict(env_prefix="PIXANO_INFERENCE_", env_file=".env", extra="ignore")

    # NoDecode: keep the raw env string (e.g. "a,b,c") out of pydantic-settings' JSON
    # decoder so the CSV validator below can split it.
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)

    media_allow_url: bool = True
    media_url_host_allowlist: Annotated[list[str], NoDecode] = Field(default_factory=list)
    media_allow_private_ips: bool = False
    media_roots: Annotated[list[Path], NoDecode] = Field(default_factory=list)
    media_connect_timeout_s: float = Field(default=5.0, gt=0)
    media_read_timeout_s: float = Field(default=30.0, gt=0)
    media_max_redirects: int = Field(default=3, ge=0)
    media_max_image_bytes: int = Field(default=50 * _MB, ge=0)
    media_max_video_bytes: int = Field(default=512 * _MB, ge=0)

    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    max_request_body_bytes: int = Field(default=100 * _MB, ge=0)

    log_level: str = "INFO"
    log_json: bool = False

    @field_validator(
        "api_keys",
        "media_url_host_allowlist",
        "media_roots",
        "cors_allow_origins",
        mode="before",
    )
    @classmethod
    def _accept_csv(cls, value: Any) -> Any:
        return _split_csv(value)

    @property
    def auth_enabled(self) -> bool:
        """Whether API-key authentication is active."""
        return len(self.api_keys) > 0
