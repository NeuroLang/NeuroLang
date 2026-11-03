"""
The LLM assistant in a web page, with the visitor's own API key.

Python in a browser (Pyodide) has no sockets: :class:`BrowserTransport`
sends the requests of an ``httpx`` client with the browser's
``XMLHttpRequest``, synchronously, as the REPL runs in a Web Worker,
where that is allowed. The page's origin is not the provider's, so the
provider must allow calls from web pages (CORS): :data:`PROVIDERS` are
those that do, and :func:`browser_config` is the configuration of the
assistant for one of them and the key of the visitor.
"""

from dataclasses import dataclass, field
from typing import List

import httpx

from .config import (
    AssistantConfig,
    LLMConfig,
    LLMConfigError,
    ProviderConfig,
)

# Headers a browser sets itself, and refuses to be set.
FORBIDDEN_HEADERS = frozenset(
    {"host", "content-length", "accept-encoding", "connection", "user-agent"}
)
# The browser has decoded the body, and httpx must not do it again.
DECODED_HEADERS = frozenset({"content-encoding", "content-length"})

BLOCKED = (
    "the browser could not send the request: there may be no network, or "
    "the provider may not allow calls from web pages (CORS)"
)


# The providers a visitor can bring a key for: those that answer calls
# from web pages (Albert, for one, does not). "custom" is any server
# speaking the OpenAI API at the address the visitor gives (e.g. Ollama).
PROVIDERS = {
    "anthropic": {
        "label": "Anthropic", "api": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-5-5", "tool_calling": "native",
        "headers": {"anthropic-dangerous-direct-browser-access": "true"},
    },
    "openai": {
        "label": "OpenAI", "api": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1", "tool_calling": "native", "headers": {},
    },
    "openrouter": {
        "label": "OpenRouter", "api": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-sonnet-5.5", "tool_calling": "native",
        "headers": {"X-Title": "NeuroLang"},
    },
    "custom": {
        "label": "Other, OpenAI-compatible", "api": "openai",
        "base_url": "", "model": "", "tool_calling": "text", "headers": {},
    },
}
DEFAULT_PROVIDER = "anthropic"


def providers_for_form() -> List[dict]:
    """Return the providers of the page's form: id, label, address, model."""
    return [
        {"id": name, "label": entry["label"], "base_url": entry["base_url"],
         "model": entry["model"]}
        for name, entry in PROVIDERS.items()
    ]

# The most rows of a result sent to the page.
MAX_ROWS = 200_000

# What the output is wrapped in: a block as wide as the terminal it
# shows, in the page's monospace font.
HTML_FORMAT = (
    '<pre class="repl-output" style="font-family: inherit">'
    "<code>{code}</code></pre>"
)


@dataclass(frozen=True)
class BrowserProvider(ProviderConfig):
    """A provider whose API key is the visitor's, given to the page."""

    key: str = field(default="", repr=False)

    def api_key(self) -> str:
        """Return the visitor's key, or say where to give it."""
        if not self.key:
            raise LLMConfigError(
                "The assistant needs your own API key: add it in the "
                "“AI assistant” panel of this page."
            )
        return self.key


def browser_config(
    provider_id: str, api_key: str = "", model: str = "", base_url: str = ""
) -> LLMConfig:
    """
    Return the configuration of the assistant using *provider_id* with
    *api_key*, its *model* and *base_url* defaulting to the provider's.

    Raise ValueError for an unknown provider, or a custom one without
    the address of its API.
    """
    if provider_id not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider_id!r}.")
    entry = PROVIDERS[provider_id]
    base_url = (base_url or entry["base_url"]).strip()
    if not base_url:
        raise ValueError("This provider needs the address of its API.")
    provider = BrowserProvider(
        name=provider_id, api=entry["api"],
        model=(model or entry["model"]).strip(), base_url=base_url,
        headers=dict(entry["headers"]), tool_calling=entry["tool_calling"],
        key=api_key.strip(),
    )
    return LLMConfig({provider_id: provider}, provider_id, AssistantConfig())


def new_xhr():
    """Return a new XMLHttpRequest of the browser."""
    import js

    return js.XMLHttpRequest.new()


def _headers(text: str) -> list:
    """Return the ``(name, value)`` headers of a response's header block."""
    headers = []
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip().lower() not in DECODED_HEADERS:
            headers.append((name.strip(), value.strip()))
    return headers


class BrowserTransport(httpx.BaseTransport):
    """
    An ``httpx`` transport sending the requests with XMLHttpRequest.

    *xhr_factory* returns the request objects, by default the browser's.
    """

    def __init__(self, xhr_factory=None):
        self._factory = xhr_factory

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Send *request* and return its response."""
        xhr = (self._factory or new_xhr)()
        xhr.open(request.method, str(request.url), False)
        for name, value in request.headers.items():
            if name.lower() not in FORBIDDEN_HEADERS:
                xhr.setRequestHeader(name, value)
        body = request.content.decode("utf-8") if request.content else None
        try:
            xhr.send(body)
        except Exception as exc:  # a JsException: network error or CORS
            raise httpx.ConnectError(f"{BLOCKED} ({exc}).", request=request)
        if not xhr.status:
            raise httpx.ConnectError(f"{BLOCKED}.", request=request)
        return httpx.Response(
            xhr.status, headers=_headers(xhr.getAllResponseHeaders()),
            content=xhr.responseText.encode("utf-8"), request=request,
        )
