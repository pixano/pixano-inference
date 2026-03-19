# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for model configuration."""

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Model Information.

    Attributes:
        name: Name of the model.
        capability: Capability of the model.
        model_path: HuggingFace repo ID or local path.
        model_class: Model class name (e.g. "Sam2ImageModel").
    """

    name: str
    capability: str
    model_path: str | None = None
    model_class: str | None = None
