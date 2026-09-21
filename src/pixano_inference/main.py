# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""CLI entrypoint for starting the Pixano Inference server."""

from pathlib import Path
from typing import Annotated, Optional

import typer


app = typer.Typer(add_completion=False)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Pixano Inference app URL host")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Pixano Inference app URL port")] = 7463,
    config: Annotated[
        Optional[Path], typer.Option(exists=True, help="Path to Python config file (.py) for model deployments")
    ] = None,
    strict_startup: Annotated[
        bool,
        typer.Option(help="Fail startup if any configured model cannot be deployed (recommended for production)."),
    ] = True,
    num_gpus: Annotated[
        Optional[int],
        typer.Option(
            min=0,
            help=(
                "Number of GPUs to give Ray. Omit to let Ray auto-detect. Set it explicitly when "
                "auto-detection is wrong -- Ray can report a GPU host as having a TPU, which makes "
                "GPU deployments fail their resource check."
            ),
        ),
    ] = None,
):
    """Start the Pixano Inference server.

    Custom models are installable packages discovered via the ``pixano_inference.models``
    entry point (``pip install`` / ``uv pip install -e`` your model package, then reference
    it by name in the config). See the custom-models docs.

    Examples:
        # Start the server
        pixano-inference --host 0.0.0.0 --port 7463

        # Start with a Python config declaring the models to deploy
        pixano-inference --host 0.0.0.0 --port 7463 --config models.py

        # Pin the GPU count when Ray's auto-detection gets it wrong
        pixano-inference --config models.py --num-gpus 1
    """
    from .observability import configure_logging
    from .ray import InferenceServer, RayServeConfig
    from .server_settings import ServerSettings

    # Structured, request-id-aware logging driven by PIXANO_INFERENCE_LOG_LEVEL / _LOG_JSON.
    settings = ServerSettings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)

    ray_config = RayServeConfig(host=host, port=port, strict_startup=strict_startup, num_gpus=num_gpus)
    server = InferenceServer(config=ray_config)

    if config is not None:
        server.register_from_config(config)

    server.start(host=host, port=port, blocking=True)
