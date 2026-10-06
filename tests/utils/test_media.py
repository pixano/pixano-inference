# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Base64 data-URI recognition for request media."""

import base64

import pytest

from pixano_inference.utils.media import (
    convert_string_video_to_bytes_or_path,
    extract_media_from_base64,
    is_base64_image,
    is_base64_video,
)


@pytest.mark.parametrize("subtype", ["mp4", "webm", "quicktime", "x-msvideo", "3gpp2"])
def test_video_data_uri_is_recognised_whatever_its_subtype(subtype):
    assert is_base64_video(f"data:video/{subtype};base64,AAAA")
    assert not is_base64_image(f"data:video/{subtype};base64,AAAA")


@pytest.mark.parametrize("subtype", ["jpeg", "png", "jp2", "svg+xml", "vnd.microsoft.icon"])
def test_image_data_uri_is_recognised_whatever_its_subtype(subtype):
    assert is_base64_image(f"data:image/{subtype};base64,AAAA")
    assert not is_base64_video(f"data:image/{subtype};base64,AAAA")


@pytest.mark.parametrize("string", ["/videos/clip.mp4", "data:video/;base64,AAAA", "video/mp4;base64,AAAA"])
def test_other_strings_are_not_data_uris(string):
    assert not is_base64_video(string)


def test_mp4_data_uri_is_decoded_to_its_bytes():
    payload = b"\x00\x00\x00\x18ftypmp42"
    data_uri = "data:video/mp4;base64," + base64.b64encode(payload).decode()

    assert extract_media_from_base64(data_uri) == base64.b64encode(payload).decode()
    assert convert_string_video_to_bytes_or_path(data_uri) == payload
