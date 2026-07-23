# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Consistent error envelope and exception handlers for the API.

Every error response has the shape ``{"error": {"code", "message", "requestId"}}``.
Unhandled exceptions never leak their text to the client; the full traceback is logged
server-side under the same request id.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)

_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    500: "internal_error",
    503: "unavailable",
    504: "timeout",
}


def _request_id(request: Request) -> str:
    # Set by RequestContextMiddleware (generated when the client sends none); fall back to the
    # raw header, then "-", so handlers still work if the middleware is absent.
    state_id = getattr(request.state, "request_id", None)
    return state_id or request.headers.get("X-Request-ID") or request.headers.get("X-Request-Id") or "-"


def _envelope(status_code: int, code: str, message, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "requestId": request_id}},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install the error-envelope exception handlers on *app*."""

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "error")
        return _envelope(exc.status_code, code, jsonable_encoder(exc.detail), _request_id(request))

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # jsonable_encoder makes validator error context (e.g. a raised ValueError) JSON-safe.
        return _envelope(422, "validation_error", jsonable_encoder(exc.errors()), _request_id(request))

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        logger.exception("Unhandled error [%s] on %s %s", request_id, request.method, request.url.path)
        return _envelope(500, "internal_error", "Internal server error.", request_id)
