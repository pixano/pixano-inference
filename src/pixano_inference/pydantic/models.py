# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for model configuration."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from pixano_inference.tasks.utils import is_task


class ModelInfo(BaseModel):
    """Model Information.

    Attributes:
        name: Name of the model.
        task: Task of the model.
    """

    name: str
    task: str


class ModelConfig(BaseModel):
    """Model configuration for instantiation.

    Attributes:
        name: Name of the model.
        task: Task of the model.
        path: Path to the model dump.
        config: Configuration of the model.
        processor_config: Configuration of the processor.
    """

    name: str
    task: str
    path: Path | str | None = None
    config: dict[str, Any] = {}
    processor_config: dict[str, Any] = {}

    @field_validator("task")
    def _check_task(cls, value):
        if not is_task(value):
            raise ValueError(f"Invalid task '{value}'")
        return value
