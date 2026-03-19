# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Concrete model implementations, organised by backend/extra.

Importing this module triggers ``@register_model`` decorators for all
available model implementations, depending on which optional dependencies
are installed.
"""

import logging


logger = logging.getLogger(__name__)

# SAM2 models ---------------------------------------------------------------
try:
    from pixano_inference.utils.package import is_sam2_installed

    if is_sam2_installed():
        from . import sam2  # noqa: F401

        logger.debug("Registered SAM2 models")
except Exception as e:
    logger.debug("SAM2 models not available: %s", e)

# SAM3 models (placeholder) ------------------------------------------------
try:
    from pixano_inference.utils.package import is_sam3_installed

    if is_sam3_installed():
        from . import sam3  # noqa: F401

        logger.debug("Registered SAM3 models")
except Exception as e:
    logger.debug("SAM3 models not available: %s", e)

# Transformers models -------------------------------------------------------
try:
    from pixano_inference.utils.package import is_transformers_installed

    if is_transformers_installed():
        from . import transformers  # noqa: F401

        logger.debug("Registered Transformers models")
except Exception as e:
    logger.debug("Transformers models not available: %s", e)

# vLLM models ---------------------------------------------------------------
try:
    from pixano_inference.utils.package import is_vllm_installed

    if is_vllm_installed():
        from . import vllm  # noqa: F401

        logger.debug("Registered vLLM models")
except Exception as e:
    logger.debug("vLLM models not available: %s", e)
