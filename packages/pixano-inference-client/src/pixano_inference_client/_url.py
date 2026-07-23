# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""URL utils."""

import re


# Only http/https are accepted. file:// and s3:// enable an SSRF / local-file-read surface when
# used to dereference client-supplied media (see the server-side media ingestion policy).
url_validation_regex = r"^https?://[^\s]+$"


def is_url(url: str) -> bool:
    """Check if a string is a valid http/https URL."""
    return re.match(url_validation_regex, url) is not None
