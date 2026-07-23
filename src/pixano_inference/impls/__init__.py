# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Concrete first-party model implementations, organised by backend/extra.

Importing this module triggers ``@register_model`` decorators for the built-in
implementations whose optional dependencies are installed. SAM2 lives in the separate
``pixano-inference-sam`` plugin package (discovered via its entry point).
"""

import logging


logger = logging.getLogger(__name__)

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
