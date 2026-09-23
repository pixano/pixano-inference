# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Check for the upstream ``sam-2`` library, which is distributed from git only."""

from pixano_inference.utils.package import assert_package_installed


def assert_sam2_installed() -> None:
    """Raise an ``ImportError`` explaining how to install ``sam-2`` when it is missing."""
    assert_package_installed(
        "sam2",
        "sam2 is not installed. Please install it using 'pip install git+https://github.com/facebookresearch/sam2.git'.",
    )
