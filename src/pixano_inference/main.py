# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Main application entry point."""

import click
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse

from . import PIXANO_INFERENCE_SETTINGS, routers


router = APIRouter()


@router.get("/", include_in_schema=False)
async def docs_redirect() -> RedirectResponse:
    """Redirect homepage to docs."""
    return RedirectResponse(url="/docs")


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        name=PIXANO_INFERENCE_SETTINGS.app_name,
        description=PIXANO_INFERENCE_SETTINGS.app_description,
        version=PIXANO_INFERENCE_SETTINGS.app_version,
    )
    app.include_router(routers.tasks.router)
    app.include_router(routers.providers.router)
    app.include_router(routers.app.router)

    app.include_router(router)
    return app


fast_api_app = create_app()


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
    default=80,
    help="Pixano Inference app URL port",
    show_default=True,
)
def serve(host: str, port: int):
    """Main application entry point."""
    config = uvicorn.Config(fast_api_app, host=host, port=port)
    server = uvicorn.Server(config)
    server.run()
