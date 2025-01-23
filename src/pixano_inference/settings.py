# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Settings for the Pixano Inference API."""

import os
from typing import Any

from pydantic_settings import BaseSettings

from pixano_inference.__version__ import __version__
from pixano_inference.utils.package import is_torch_installed


if is_torch_installed():
    import torch


class Settings(BaseSettings):
    """Application settings."""

    app_name: str = "Pixano Inference"
    app_version: str = __version__
    app_description: str = "Pixano Inference API for multimodal tasks."
    num_cpus: int
    num_gpus: int
    num_nodes: int = 1  # TODO: for now only single node is supported
    gpus_used: list[int] = []

    def __init__(self, **data: Any):
        """Initialize the settings."""
        if "num_cpus" not in data:
            data["num_cpus"] = os.cpu_count()
        if "num_gpus" not in data:
            if is_torch_installed():
                if torch.cuda.is_available():
                    data["num_gpus"] = torch.cuda.device_count()
                else:
                    data["num_gpus"] = 0
            else:
                data["num_gpus"] = 0

        super().__init__(**data)

    @property
    def gpus_available(self) -> list[int]:
        """Return the available GPUs."""
        return [i for i in range(self.num_gpus) if i not in self.gpus_used]


def get_pixano_inference_settings() -> Settings:
    """Return the settings."""
    return PIXANO_INFERENCE_SETTINGS


PIXANO_INFERENCE_SETTINGS = Settings()
