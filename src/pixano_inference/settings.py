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
from pixano_inference.utils.package import is_torch_installed


load_dotenv()


if is_torch_installed():
    import torch


class Settings(BaseSettings):
    """Application settings.

    Attributes:
        app_name: The name of the application.
        app_version: The version of the application.
        app_description: A description of the application.
        num_cpus: The number of CPUs accessible to the application.
        num_gpus: The number of GPUs available for inference.
        num_nodes: The number of nodes available for inference.
        gpus_used: The list of GPUs used by the application.
    """

    app_name: str = "Pixano Inference"
    app_version: str = __version__
    app_description: str = "Pixano Inference API for multimodal tasks."
    num_cpus: int
    num_gpus: int
    num_nodes: int = 1  # TODO: for now only single node is supported
    gpus_used: list[int] = []
    gpu_to_model: dict[int, str] = {}
    models: list[str] = []
    models_to_task: dict[str, str] = {}

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

    def reserve_gpu(self) -> int | None:
        """Reserve a gpu if any available."""
        gpus = self.gpus_available
        if len(gpus) == 0:
            return None
        selected_gpu = gpus[0]
        self.gpus_used.append(selected_gpu)
        return selected_gpu

    def free_gpu(self, gpu: int) -> None:
        """Free a GPU if used."""
        try:
            self.gpus_used.remove(gpu)
        except ValueError:
            pass
        self.gpu_to_model.pop(gpu)
        return

    def add_model(self, model: str, task: str) -> int | None:
        """Add a model."""
        if model in self.models:
            raise ValueError(f"Model {model} already registered.")
        gpu = self.assign_model_gpu(model)
        self.models_to_task[model] = task
        self.models.append(model)
        return gpu

    def assign_model_gpu(self, model: str) -> int | None:
        """Assign a model to a gpu.

        Args:
            model: The model name.

        Returns:
            The gpu index. If no gpu available, returns None.
        """
        gpu = self.reserve_gpu()
        if gpu is not None:
            self.gpu_to_model[gpu] = model
        return gpu

    def remove_model(self, model: str) -> None:
        """Remove a model."""
        self.models.remove(model)
        self.models_to_task.pop(model, None)
        for gpu, model_stored in self.gpu_to_model.items():
            if model_stored == model:
                self.free_gpu(gpu)
                break
        return


def get_pixano_inference_settings() -> Settings:
    """Return the settings."""
    return PIXANO_INFERENCE_SETTINGS


PIXANO_INFERENCE_SETTINGS = Settings()
