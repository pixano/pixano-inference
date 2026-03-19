# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""CLI entrypoint for starting the Pixano Inference server."""

import sys
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
    module_path: Annotated[
        Optional[list[Path]],
        typer.Option(help="Directory to add to Python path for custom model modules. Can be repeated."),
    ] = None,
):
    """Start the Pixano Inference server.

    Examples:
        # Start the server
        pixano-inference --host 0.0.0.0 --port 7463

        # Start with Python config
        pixano-inference --host 0.0.0.0 --port 7463 --config models.py

        # Start with custom model modules
        pixano-inference --module-path /path/to/my-models --config my_config.py
    """
    if module_path:
        for p in module_path:
            resolved = str(p.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)

    from .ray import InferenceServer, RayServeConfig

    ray_config = RayServeConfig(host=host, port=port)
    server = InferenceServer(config=ray_config)

    if config is not None:
        server.register_from_config(config)

    server.start(host=host, port=port, blocking=True)
