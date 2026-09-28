"""JevClient over the official TypeSafe SDK (install the ``typesafe`` extra). Always the latest Jev.

Hosts with their own runtime (for example one that routes requests by region) pass an adapter over
that instead; the library only needs ``ask`` and ``model``.

When this adapter builds the SDK client itself, it wraps the HTTP transport so ``send`` returns the
exact response bytes, status and content type for a journal. A caller-supplied SDK client is not
wrapped; its responses are journaled decoded and marked inexact.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

from ..judgments.answers import JevResponse, response_from_raw
from ..judgments.client import LATEST_JEV
from ..judgments.journal import RawResponse


class CapturingTransport:
    """Passes requests to ``inner`` and keeps each thread's last response, with the exact request
    body that produced it, as a ``RawResponse``."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._last = threading.local()

    def handle_request(self, request):
        response = self._inner.handle_request(request)
        response.read()
        content_type = response.headers.get("content-type", "")
        self._last.response = RawResponse(
            response.content, response.status_code, content_type, sent_body=request.content
        )
        return response

    def take(self) -> RawResponse | None:
        captured = getattr(self._last, "response", None)
        self._last.response = None
        return captured

    def close(self) -> None:
        self._inner.close()


class TypeSafeJevClient:
    def __init__(self, sdk_client=None, model: str = LATEST_JEV, *, transport=None) -> None:
        """``transport`` is the inner HTTP transport to wrap (default: the SDK's standard one)."""
        self._capture: CapturingTransport | None = None
        if sdk_client is None:
            import httpx2
            from typesafe_sdk import TypeSafeClient

            self._capture = CapturingTransport(transport or httpx2.HTTPTransport())
            sdk_client = TypeSafeClient(model=model, transport=self._capture)
        self._sdk = sdk_client
        self.model = model

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        return self.parse(self.send(state, questions))

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        """The response as received, with the SDK's decoded form attached."""
        response = self._sdk.system_one(dict(state), dict(questions))
        decoded = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
        captured = self._capture.take() if self._capture else None
        if captured is None:
            return RawResponse.from_decoded(decoded)
        return RawResponse(
            captured.body, captured.status, captured.content_type, decoded, sent_body=captured.sent_body
        )

    def parse(self, raw: RawResponse) -> JevResponse:
        return response_from_raw(raw.json())
