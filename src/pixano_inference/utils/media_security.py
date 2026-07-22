# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Security policy for media ingestion (SSRF guard, size/time limits, path containment).

Client requests reference images and video by URL, local path, or base64. Dereferencing
those references is a classic SSRF and arbitrary-file-read surface. Every fetch and every
local-path resolution goes through :class:`MediaPolicy` here, which:

* accepts only ``http``/``https`` URLs (never ``file://``/``s3://``);
* resolves the URL host and rejects private/loopback/link-local/reserved IPs (with an
  optional host allowlist for trusted internal services);
* re-validates every redirect hop;
* enforces connect/read timeouts and a streamed maximum-byte cap;
* denies local-path media unless the path resolves under a configured ``media_roots`` entry.

The active policy is a per-process global built lazily from :class:`ServerSettings`, so a
Ray worker reconstructs the same policy from the environment its driver passed on.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit


if TYPE_CHECKING:
    import requests

    from pixano_inference.server_settings import ServerSettings


class MediaSecurityError(ValueError):
    """Raised when a media reference violates the security policy."""


@dataclass
class MediaPolicy:
    """Resolved media-ingestion security policy for the current process."""

    allow_url: bool = True
    url_host_allowlist: frozenset[str] = frozenset()
    allow_private_ips: bool = False
    media_roots: tuple[Path, ...] = ()
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 30.0
    max_redirects: int = 3
    max_image_bytes: int = 50 * 1024 * 1024
    max_video_bytes: int = 512 * 1024 * 1024

    @classmethod
    def from_settings(cls, settings: ServerSettings) -> MediaPolicy:
        """Build a policy from :class:`ServerSettings`."""
        return cls(
            allow_url=settings.media_allow_url,
            url_host_allowlist=frozenset(h.lower() for h in settings.media_url_host_allowlist),
            allow_private_ips=settings.media_allow_private_ips,
            media_roots=tuple(Path(p).expanduser().resolve() for p in settings.media_roots),
            connect_timeout_s=settings.media_connect_timeout_s,
            read_timeout_s=settings.media_read_timeout_s,
            max_redirects=settings.media_max_redirects,
            max_image_bytes=settings.media_max_image_bytes,
            max_video_bytes=settings.media_max_video_bytes,
        )

    @classmethod
    def from_env(cls) -> MediaPolicy:
        """Build a policy from the environment (via :class:`ServerSettings`)."""
        from pixano_inference.server_settings import ServerSettings

        return cls.from_settings(ServerSettings())


_ACTIVE_POLICY: MediaPolicy | None = None


def set_media_policy(policy: MediaPolicy) -> None:
    """Install the active media policy for this process."""
    global _ACTIVE_POLICY
    _ACTIVE_POLICY = policy


def get_media_policy() -> MediaPolicy:
    """Return the active media policy, lazily building a secure default from the env."""
    global _ACTIVE_POLICY
    if _ACTIVE_POLICY is None:
        _ACTIVE_POLICY = MediaPolicy.from_env()
    return _ACTIVE_POLICY


def is_http_url(value: str) -> bool:
    """Whether *value* is an ``http``/``https`` URL (the only fetchable schemes)."""
    scheme = urlsplit(value).scheme.lower()
    return scheme in {"http", "https"}


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def _assert_host_allowed(host: str, policy: MediaPolicy) -> None:
    """Reject a host whose resolved IPs are private/loopback/etc., unless allowlisted."""
    host_l = host.lower()
    if host_l in policy.url_host_allowlist:
        return
    if policy.allow_private_ips:
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise MediaSecurityError(f"Could not resolve media host '{host}'.") from exc

    for info in infos:
        raw_addr = info[4][0]
        if not isinstance(raw_addr, str):
            continue
        # Strip IPv6 zone id if present.
        addr = raw_addr.split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _ip_is_blocked(ip):
            raise MediaSecurityError(
                f"Media host '{host}' resolves to a blocked address ({addr}). "
                "Set PIXANO_INFERENCE_MEDIA_URL_HOST_ALLOWLIST to permit trusted internal hosts."
            )


def _validate_url(url: str, policy: MediaPolicy) -> None:
    if not policy.allow_url:
        raise MediaSecurityError("URL media ingestion is disabled by policy.")
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"}:
        raise MediaSecurityError(f"Only http/https media URLs are allowed, got scheme '{parts.scheme}'.")
    if not parts.hostname:
        raise MediaSecurityError("Media URL has no host.")
    _assert_host_allowed(parts.hostname, policy)


def fetch_url_bytes(url: str, *, max_bytes: int, policy: MediaPolicy | None = None) -> bytes:
    """Fetch a URL as bytes under the media policy (SSRF-guarded, size/time-capped).

    Args:
        url: The http/https URL to fetch.
        max_bytes: Maximum number of bytes to read before aborting.
        policy: Policy to apply; defaults to the active process policy.

    Returns:
        The response body bytes.

    Raises:
        MediaSecurityError: On any policy violation (bad scheme/host, too many redirects,
            or the response exceeding *max_bytes*).
    """
    import requests

    policy = policy or get_media_policy()
    timeout = (policy.connect_timeout_s, policy.read_timeout_s)

    current = url
    with requests.Session() as session:
        for _ in range(policy.max_redirects + 1):
            _validate_url(current, policy)
            response = session.get(current, stream=True, allow_redirects=False, timeout=timeout)
            try:
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        raise MediaSecurityError("Redirect response without a Location header.")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                return _read_capped(response, max_bytes)
            finally:
                response.close()
    raise MediaSecurityError(f"Too many redirects while fetching media (> {policy.max_redirects}).")


def _read_capped(response: requests.Response, max_bytes: int) -> bytes:
    """Read a streamed response body, aborting if it exceeds *max_bytes*."""
    declared = response.headers.get("Content-Length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                raise MediaSecurityError(f"Media exceeds the maximum allowed size ({max_bytes} bytes).")
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=1 << 16):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise MediaSecurityError(f"Media exceeds the maximum allowed size ({max_bytes} bytes).")
        chunks.append(chunk)
    return b"".join(chunks)


def resolve_local_path(raw_path: str | Path, policy: MediaPolicy | None = None) -> Path:
    """Resolve a client-supplied local path, enforcing containment under ``media_roots``.

    Args:
        raw_path: The path from the request.
        policy: Policy to apply; defaults to the active process policy.

    Returns:
        The resolved, real path.

    Raises:
        MediaSecurityError: If local-path media is disabled (no roots configured) or the
            path escapes every configured root.
    """
    policy = policy or get_media_policy()
    if not policy.media_roots:
        raise MediaSecurityError("Local-path media is disabled. Configure PIXANO_INFERENCE_MEDIA_ROOTS to allow it.")
    resolved = Path(raw_path).expanduser().resolve()
    for root in policy.media_roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise MediaSecurityError(f"Path '{raw_path}' is outside the allowed media roots.")
