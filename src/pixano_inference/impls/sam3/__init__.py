# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""SAM3 model implementations -- placeholder.

This sub-package is a template for adding new model backends. To add a
SAM3 model:

1. Create a new module (e.g. ``image.py``) in this directory.
2. Define a class inheriting from the appropriate base in
   ``pixano_inference.models`` (e.g. ``SegmentationModel``).
3. Decorate it with ``@register_model("Sam3ImageModel")``.
4. Import it in this ``__init__.py`` so the decorator fires on import.
5. Add any required dependencies to the ``[sam3]`` extra in
   ``pyproject.toml``.
"""
