"""Tests of the HTTP transport of the LLM clients in a browser."""

import sys

import httpx
import pytest

from ..browser import BrowserTransport
from ..config import ProviderConfig
from ..providers import LLMError, make_client
from ..validation import _in_child, _with_timeout


class FakeXhr:
    """A synchronous XMLHttpRequest answering with canned values."""

    sent = []

    def __init__(self, status=200, text='{"ok": true}', headers=""):
        self.status = status
        self.responseText = text
        self._headers = headers
        self.request_headers = {}
        self.timeout = 0

    def open(self, method, url, asynchronous):
        self.call = {"method": method, "url": url, "async": asynchronous}

    def setRequestHeader(self, name, value):
        self.request_headers[name] = value

    def send(self, body=None):
        self.call["body"] = body
        FakeXhr.sent.append(self)

    def getAllResponseHeaders(self):
        return self._headers


def transport(**answer):
    FakeXhr.sent = []
    return BrowserTransport(lambda: FakeXhr(**answer))


def test_a_request_is_sent_synchronously_and_its_answer_returned():
    client = httpx.Client(transport=transport(
        status=201, text='{"a": 1}',
        headers="content-type: application/json\r\nx-request-id: r1\r\n",
    ))
    response = client.post(
        "https://llm.test/v1/messages", json={"q": 1},
        headers={"x-api-key": "secret", "anthropic-version": "2023-06-01"},
    )
    [xhr] = FakeXhr.sent
    assert xhr.call["method"] == "POST"
    assert xhr.call["url"] == "https://llm.test/v1/messages"
    assert xhr.call["async"] is False
    assert xhr.call["body"] == '{"q":1}' or xhr.call["body"] == '{"q": 1}'
    assert xhr.request_headers["x-api-key"] == "secret"
    assert xhr.request_headers["content-type"] == "application/json"
    assert response.status_code == 201
    assert response.json() == {"a": 1}
    assert response.headers["x-request-id"] == "r1"


def test_headers_the_browser_forbids_are_not_set():
    client = httpx.Client(transport=transport())
    client.post("https://llm.test/v1/x", json={})
    [xhr] = FakeXhr.sent
    forbidden = {"host", "content-length", "accept-encoding", "connection",
                 "user-agent"}
    assert not forbidden & {name.lower() for name in xhr.request_headers}


def test_a_blocked_request_says_it_may_be_the_providers_cors():
    class Blocked(FakeXhr):
        def send(self, body=None):
            raise RuntimeError("NetworkError: Failed to execute 'send'")

    client = httpx.Client(transport=BrowserTransport(Blocked))
    with pytest.raises(httpx.ConnectError, match="CORS"):
        client.post("https://llm.test/v1/x", json={})


def test_a_provider_of_the_browser_uses_the_transport(monkeypatch):
    monkeypatch.setattr(sys, "platform", "emscripten")
    monkeypatch.setattr(
        "neurolang_apps.utils.llm.browser.new_xhr",
        lambda: FakeXhr(text='{"choices": [{"message": {"content": "hi"}}]}'),
    )
    provider = ProviderConfig("p", "openai", "m", "https://llm.test/v1")
    assert make_client(provider).complete(
        [{"role": "user", "content": "q"}]
    ).text == "hi"


def test_a_provider_error_in_the_browser_is_an_llm_error(monkeypatch):
    monkeypatch.setattr(sys, "platform", "emscripten")
    monkeypatch.setattr(
        "neurolang_apps.utils.llm.browser.new_xhr",
        lambda: FakeXhr(status=401, text="bad key"),
    )
    provider = ProviderConfig("p", "openai", "m", "https://llm.test/v1")
    with pytest.raises(LLMError, match="HTTP 401: bad key"):
        make_client(provider).complete([{"role": "user", "content": "q"}])


def test_queries_run_inline_where_there_are_no_threads_nor_processes(
    monkeypatch,
):
    monkeypatch.setattr(sys, "platform", "emscripten")

    def no_threads(*args, **kwargs):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr("threading.Thread.start", no_threads)
    assert _with_timeout(lambda: 42, 5.0) == 42
    assert _in_child(lambda: 43, 5.0, 1024.0) == 43
