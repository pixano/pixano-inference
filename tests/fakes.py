# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Awaitable stand-ins for Ray Serve deployment handles/responses, for unit tests."""

from __future__ import annotations


class FakeResponse:
    """Awaitable stand-in for a Serve DeploymentResponse.

    Resolves immediately by default; ``pending=True`` never resolves until cancelled (used
    to test the async job cancel path without a completion race).
    """

    def __init__(self, result, *, error: Exception | None = None, pending: bool = False):
        self._result = result
        self._error = error
        self._pending = pending
        self.cancelled = False

    def __await__(self):
        import asyncio

        async def _resolve():
            if self._pending:
                await asyncio.Event().wait()  # never set; cancelled via task.cancel()
            if self._error is not None:
                raise self._error
            return self._result

        return _resolve().__await__()

    def cancel(self):
        self.cancelled = True


class FakeRemoteMethod:
    def __init__(self, result, *, error: Exception | None = None, pending: bool = False):
        self._result = result
        self._error = error
        self._pending = pending
        self.last_input = None
        self.last_response: FakeResponse | None = None

    def remote(self, input_data):
        self.last_input = input_data
        self.last_response = FakeResponse(self._result, error=self._error, pending=self._pending)
        return self.last_response


class FakeHandle:
    def __init__(self, result, *, error: Exception | None = None, pending: bool = False):
        self.predict = FakeRemoteMethod(result, error=error, pending=pending)
