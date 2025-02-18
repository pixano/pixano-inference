# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""URL utils."""

import re


# TODO: improve this regex pattern
url_validation_regex = r"^(file://|https?://|s3://)[^\s]+$"


def is_url(url: str) -> bool:
    """Check if a string is a valid URL."""
    return re.match(url_validation_regex, url) is not None
