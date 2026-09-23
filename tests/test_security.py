# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for API-key auth and the media-ingestion security policy."""

from pathlib import Path

import pytest

from pixano_inference.security import _extract_key, _key_matches
from pixano_inference.server_settings import ServerSettings
from pixano_inference.utils.media_security import (
    MediaPolicy,
    MediaSecurityError,
    is_http_url,
    resolve_local_path,
)


# --- ServerSettings parsing ---------------------------------------------------------


def test_api_keys_parsed_from_csv(monkeypatch):
    monkeypatch.setenv("PIXANO_INFERENCE_API_KEYS", "key-a, key-b ,key-c")
    settings = ServerSettings()
    assert settings.api_keys == ["key-a", "key-b", "key-c"]
    assert settings.auth_enabled is True


def test_auth_disabled_when_no_keys(monkeypatch):
    monkeypatch.delenv("PIXANO_INFERENCE_API_KEYS", raising=False)
    settings = ServerSettings()
    assert settings.api_keys == []
    assert settings.auth_enabled is False


# --- API-key matching ---------------------------------------------------------------


def test_extract_key_prefers_x_api_key():
    assert _extract_key("secret", None) == "secret"
    assert _extract_key(None, "Bearer secret") == "secret"
    assert _extract_key(None, "bearer secret") == "secret"
    assert _extract_key(None, "Basic secret") is None
    assert _extract_key(None, None) is None


def test_key_matches_is_membership():
    assert _key_matches("b", ["a", "b", "c"]) is True
    assert _key_matches("z", ["a", "b", "c"]) is False
    assert _key_matches("a", []) is False


# --- Media URL policy (SSRF guard) --------------------------------------------------


def test_is_http_url_rejects_non_http_schemes():
    assert is_http_url("http://example.com/x.jpg")
    assert is_http_url("https://example.com/x.jpg")
    assert not is_http_url("file:///etc/passwd")
    assert not is_http_url("s3://bucket/key")
    assert not is_http_url("data:image/png;base64,AAAA")
    assert not is_http_url("/local/path.jpg")


def test_fetch_rejects_file_scheme():
    from pixano_inference.utils.media_security import fetch_url_bytes

    policy = MediaPolicy()
    with pytest.raises(MediaSecurityError):
        fetch_url_bytes("file:///etc/passwd", max_bytes=1024, policy=policy)


def test_fetch_blocks_private_and_metadata_hosts():
    from pixano_inference.utils.media_security import fetch_url_bytes

    policy = MediaPolicy(allow_private_ips=False)
    for url in (
        "http://127.0.0.1/x.jpg",
        "http://localhost/x.jpg",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/x.jpg",
        "http://192.168.1.1/x.jpg",
    ):
        with pytest.raises(MediaSecurityError):
            fetch_url_bytes(url, max_bytes=1024, policy=policy)


def test_private_ip_allowed_when_host_allowlisted():
    from pixano_inference.utils import media_security

    policy = MediaPolicy(url_host_allowlist=frozenset({"localhost"}))
    # Host is allowlisted, so the SSRF guard passes; the request itself will fail to
    # connect in the test env, which is a different (non-security) error.
    media_security._validate_url("http://localhost/x.jpg", policy)  # no MediaSecurityError


# --- Local-path policy (arbitrary file read) ----------------------------------------


def test_local_path_denied_without_roots():
    policy = MediaPolicy(media_roots=())
    with pytest.raises(MediaSecurityError):
        resolve_local_path("/etc/passwd", policy)


def test_local_path_escape_blocked(tmp_path: Path):
    root = tmp_path / "media"
    root.mkdir()
    (root / "ok.jpg").write_bytes(b"x")
    policy = MediaPolicy(media_roots=(root.resolve(),))

    # A path inside the root resolves.
    assert resolve_local_path(str(root / "ok.jpg"), policy) == (root / "ok.jpg").resolve()

    # Traversal outside the root is rejected.
    with pytest.raises(MediaSecurityError):
        resolve_local_path(str(root / ".." / "secret.txt"), policy)
    with pytest.raises(MediaSecurityError):
        resolve_local_path("/etc/passwd", policy)


def test_max_bytes_cap_enforced(monkeypatch):
    """A response exceeding max_bytes is rejected even without a Content-Length header."""
    from pixano_inference.utils import media_security

    class _FakeResponse:
        is_redirect = False
        is_permanent_redirect = False
        headers: dict = {}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"a" * 2048

        def close(self):
            return None

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            return _FakeResponse()

    monkeypatch.setattr("requests.Session", _FakeSession)
    policy = MediaPolicy(url_host_allowlist=frozenset({"example.com"}))
    with pytest.raises(MediaSecurityError):
        media_security.fetch_url_bytes("http://example.com/big.bin", max_bytes=1024, policy=policy)


# --- End-to-end auth enforcement on the app -----------------------------------------


def test_inference_routes_enforce_api_key(monkeypatch):
    """With keys configured, inference routes require a valid key; /health stays open."""
    pytest.importorskip("torch")  # create_ray_serve_app imports the model backends
    from fastapi.testclient import TestClient

    from pixano_inference.ray.app import create_ray_serve_app
    from pixano_inference.ray.config import RayServeConfig

    monkeypatch.setenv("PIXANO_INFERENCE_API_KEYS", "topsecret")
    app, _ = create_ray_serve_app(RayServeConfig(num_gpus=0))
    client = TestClient(app, raise_server_exceptions=False)

    payload = {"model": "nope", "image": "https://example.com/x.jpg"}

    # No key -> 401 (auth runs before the endpoint body).
    assert client.post("/v1/inference/detection", json=payload).status_code == 401
    # Wrong key -> 403.
    assert client.post("/v1/inference/detection", json=payload, headers={"X-API-Key": "wrong"}).status_code == 403
    # Correct key -> passes auth (downstream may 404/422, but never 401/403).
    ok = client.post("/v1/inference/detection", json=payload, headers={"X-API-Key": "topsecret"})
    assert ok.status_code not in (401, 403)
    # Bearer form also works.
    ok_bearer = client.post("/v1/inference/detection", json=payload, headers={"Authorization": "Bearer topsecret"})
    assert ok_bearer.status_code not in (401, 403)
    # Health probe stays unauthenticated.
    assert client.get("/health").status_code == 200


def test_inference_routes_open_when_no_keys(monkeypatch):
    """Without keys, auth is a no-op (routes reachable, downstream handles them)."""
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from pixano_inference.ray.app import create_ray_serve_app
    from pixano_inference.ray.config import RayServeConfig

    monkeypatch.delenv("PIXANO_INFERENCE_API_KEYS", raising=False)
    app, _ = create_ray_serve_app(RayServeConfig(num_gpus=0))
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/v1/inference/detection", json={"model": "nope", "image": "https://example.com/x.jpg"})
    assert resp.status_code not in (401, 403)


# --- Image decode hardening ---------------------------------------------------------


def _png_bytes(size=(4, 4)):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def _set_policy(monkeypatch, **kwargs):
    from pixano_inference.utils import media_security

    policy = MediaPolicy(**kwargs)
    monkeypatch.setattr(media_security, "get_media_policy", lambda: policy)
    monkeypatch.setattr("pixano_inference.utils.media.get_media_policy", lambda: policy)
    return policy


def test_allowed_image_format_decodes(monkeypatch):
    from pixano_inference.utils.media import convert_string_to_image

    _set_policy(monkeypatch, allowed_image_formats=frozenset({"PNG"}))
    image = convert_string_to_image(_png_bytes())
    assert image.size == (4, 4)


def test_disallowed_image_format_is_rejected(monkeypatch):
    """Keeps request bytes away from Pillow's rarely-used (and buggier) decoders."""
    from pixano_inference.utils.media import convert_string_to_image

    _set_policy(monkeypatch, allowed_image_formats=frozenset({"JPEG"}))
    with pytest.raises(ValueError, match="format PNG is not allowed"):
        convert_string_to_image(_png_bytes())


def test_empty_allowlist_permits_any_format(monkeypatch):
    from pixano_inference.utils.media import convert_string_to_image

    _set_policy(monkeypatch, allowed_image_formats=frozenset())
    assert convert_string_to_image(_png_bytes()).size == (4, 4)


def test_pixel_cap_rejects_decompression_bomb(monkeypatch):
    """A small file can decode to a huge image, so the byte caps do not cover this."""
    from pixano_inference.utils.media import convert_string_to_image

    payload = _png_bytes((2000, 2000))
    assert len(payload) < 100_000, "test payload should be small relative to its pixel count"
    _set_policy(monkeypatch, max_image_pixels=1000, allowed_image_formats=frozenset({"PNG"}))
    with pytest.raises(ValueError, match="too large to decode"):
        convert_string_to_image(payload)


def test_pixel_cap_of_zero_disables_the_check(monkeypatch):
    from pixano_inference.utils.media import convert_string_to_image

    _set_policy(monkeypatch, max_image_pixels=0, allowed_image_formats=frozenset({"PNG"}))
    assert convert_string_to_image(_png_bytes((50, 50))).size == (50, 50)


def test_media_settings_reach_the_policy(monkeypatch):
    monkeypatch.setenv("PIXANO_INFERENCE_MEDIA_MAX_IMAGE_PIXELS", "4242")
    monkeypatch.setenv("PIXANO_INFERENCE_MEDIA_ALLOWED_IMAGE_FORMATS", "png, jpeg")
    policy = MediaPolicy.from_settings(ServerSettings())
    assert policy.max_image_pixels == 4242
    assert policy.allowed_image_formats == frozenset({"PNG", "JPEG"})
