# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Main application entry point."""

import click
import uvicorn
from fastapi import FastAPI

from . import routers


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI()
    app.include_router(routers.image.router)
    app.include_router(routers.nlp.router)
    app.include_router(routers.multimodal.router)
    app.include_router(routers.providers.router)
    app.include_router(routers.video.router)
    return app


@click.command(context_settings={"auto_envvar_prefix": "UVICORN"})
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Pixano Inference app URL host",
    show_default=True,
)
@click.option(
    "--port",
    type=int,
    default=0,
    help="Pixano Inference app URL port",
    show_default=True,
)
def serve(host: str, port: int):
    """Main application entry point."""
    app = create_app()
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    server.run()
