# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Settings for the Pixano Inference API."""

import os
from typing import Any

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from pixano_inference.__version__ import __version__
from pixano_inference.utils.gpu import detect_num_gpus


load_dotenv()


class Settings(BaseSettings):
    """Application settings.

    Attributes:
        app_name: The name of the application.
        app_version: The version of the application.
        app_description: A description of the application.
        num_cpus: The number of CPUs accessible to the application.
        num_gpus: The number of GPUs available for inference.
        num_nodes: The number of nodes available for inference.
    """

    app_name: str = "Pixano Inference"
    app_version: str = __version__
    app_description: str = "Pixano Inference API for multimodal tasks."
    num_cpus: int
    num_gpus: int
    num_nodes: int = 1
    gpus_used: float = 0.0
    gpu_to_model: dict[int, str] = {}
    models: list[str] = []
    models_to_task: dict[str, str] = {}

    def __init__(self, **data: Any):
        """Initialize the settings."""
        if "num_cpus" not in data:
            data["num_cpus"] = os.cpu_count()
        if "num_gpus" not in data:
            data["num_gpus"] = detect_num_gpus()

        super().__init__(**data)
