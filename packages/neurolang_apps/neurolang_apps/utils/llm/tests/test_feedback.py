"""Tests of the sharing of the assistant's answers with a feedback server."""

import json
from collections import Counter
from pathlib import Path

import httpx
import pytest

from ..agent import AgentResult, AgentStep
from ..config import ProviderConfig, parse_config, LLMConfigError
from ..feedback import FEEDBACK_URL_ENV, FeedbackError, feedback_client

PROVIDER = ProviderConfig(
    "secret-provider", "openai", "m", "http://internal.host/v1",
    api_key_env="SECRET_KEY_ENV", headers={"X-Secret": "header-value"},
)
QUERY = "ans(s) :- mentions(s, 'pain')"


def config(**feedback):
    data = {"providers": {"p": {"base_url": "http://x/v1", "model": "m"}}}
    if feedback:
        data["feedback"] = feedback
    return parse_config(data, Path("/home/someone/llm.yaml"))


def result():
    return AgentResult(
        query=QUERY, language="datalog", valid=True, calls=2,
        usage=Counter(input=10, output=5), seconds=1.25,
        explanation=f"Read from {Path.home()}/data",
        steps=[AgentStep("t", "run_query", {"query": QUERY}, "s\n1"),
               AgentStep("", "final_answer", {"query": QUERY}, "Accepted.")],
    )


def recording(status=201):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status, json={"id": "a" * 32})
    return requests, httpx.MockTransport(handle)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(FEEDBACK_URL_ENV, raising=False)
    assert feedback_client(config()) is None
    assert feedback_client(config(url="http://fb.test")) is None


def test_enabled_by_the_environment(monkeypatch):
    monkeypatch.setenv(FEEDBACK_URL_ENV, "http://env.test")
    assert feedback_client(config()).url == "http://env.test"


def test_unknown_feedback_setting():
    with pytest.raises(LLMConfigError, match="feedback"):
        config(enabled=True, token="x")


def test_shares_the_answer_without_secrets(monkeypatch):
    monkeypatch.setenv("SECRET_KEY_ENV", "sk-very-secret")
    monkeypatch.delenv(FEEDBACK_URL_ENV, raising=False)
    requests, transport = recording()
    said = []
    client = feedback_client(
        config(enabled=True, url="http://fb.test", user="ann"),
        said.append, transport,
    )
    for _ in range(2):
        assert client.share(
            "Pain?", result(), PROVIDER, "react", "neurosynth"
        ) == "a" * 32
    assert len(said) == 1 and "http://fb.test" in said[0]
    assert [str(r.url) for r in requests] == ["http://fb.test/api/answers"] * 2
    body = requests[0].content.decode()
    for secret in (
        "sk-very-secret", "SECRET_KEY_ENV", "header-value", "internal.host",
        "/home/someone", str(Path.home()),
    ):
        assert secret not in body
    record = json.loads(body)
    assert record["question"] == "Pain?" and record["query"] == QUERY
    assert record["rows"] == "s\n1" and record["tokens"] == 15
    assert record["provider"] == "secret-provider" and record["user"] == "ann"
    assert record["explanation"] == "Read from ~/data"
    assert record["steps"][0]["action"] == "run_query"


def test_failure_warns_once():
    said = []
    _, transport = recording(status=500)
    client = feedback_client(
        config(enabled=True, url="http://fb.test"), said.append, transport
    )
    assert client.share("Q?", result(), PROVIDER, "icl") is None
    assert client.share("Q?", result(), PROVIDER, "icl") is None
    assert len(said) == 2 and "not shared" in said[1]


def test_rate_posts_the_corrected_query():
    requests, transport = recording()
    client = feedback_client(
        config(enabled=True, url="http://fb.test/", user="bob"),
        print, transport,
    )
    client.rate("b" * 32, "incorrect", "too broad", "ans(s) :- study(s)",
                "s\n1\n2")
    assert str(requests[0].url) == (
        f"http://fb.test/api/answers/{'b' * 32}/feedback"
    )
    assert json.loads(requests[0].content) == {
        "verdict": "incorrect", "user": "bob", "comment": "too broad",
        "corrected_query": "ans(s) :- study(s)",
        "corrected_query_checked": True, "corrected_rows": "s\n1\n2",
    }
    _, failing = recording(status=404)
    client = feedback_client(
        config(enabled=True, url="http://fb.test"), print, failing
    )
    with pytest.raises(FeedbackError):
        client.rate("c" * 32, "correct")


GRIST = dict(
    enabled=True, url="https://grist.test", grist_doc="doc1", user="ann"
)
RECORDS = "https://grist.test/api/docs/doc1/tables/Answers/records"


def grist(monkeypatch, said=None):
    monkeypatch.setenv("GRIST_API_KEY", "grist-secret")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"records": [{"id": 7}]})
    client = feedback_client(
        config(**GRIST), (said if said is not None else []).append,
        httpx.MockTransport(handle),
    )
    return requests, client


def test_grist_shares_a_row(monkeypatch):
    said = []
    requests, client = grist(monkeypatch, said)
    assert client.share("Pain?", result(), PROVIDER, "react", "ns") == "7"
    (request,) = requests
    assert request.method == "POST" and str(request.url) == RECORDS
    assert request.headers["Authorization"] == "Bearer grist-secret"
    (row,) = json.loads(request.content)["records"]
    fields = row["fields"]
    assert fields["question"] == "Pain?" and fields["valid"] is True
    assert json.loads(fields["steps"])[0]["action"] == "run_query"
    assert isinstance(fields["time"], float) and fields["user"] == "ann"
    assert "https://grist.test/doc1" in said[0]
    assert not any("grist-secret" in text for text in said)


def test_grist_shares_failures(monkeypatch):
    requests, client = grist(monkeypatch)
    failed = AgentResult(language="datalog", error="HTTP 429")
    assert client.share("Pain?", failed, PROVIDER, "react") == "7"
    fields = json.loads(requests[0].content)["records"][0]["fields"]
    assert fields["valid"] is False and fields["error"] == "HTTP 429"
    assert fields["query"] == "" and fields["steps"] == "[]"


def test_grist_rate_patches_the_row(monkeypatch):
    requests, client = grist(monkeypatch)
    assert client.rate("7", "incorrect", "too broad") == "7"
    client.rate("7", "unsure", "", "ans(s) :- study(s)", "s\n1")
    assert [r.method for r in requests] == ["PATCH"] * 2
    assert {str(r.url) for r in requests} == {RECORDS}
    assert [json.loads(r.content) for r in requests] == [
        {"records": [{"id": 7, "fields": {
            "verdict": "bad", "comment": "too broad"}}]},
        {"records": [{"id": 7, "fields": {
            "verdict": "unsure", "comment": "",
            "corrected_query": "ans(s) :- study(s)"}}]},
    ]
    with pytest.raises(FeedbackError, match="not a row"):
        client.rate("a" * 32, "correct")


def test_grist_without_key_warns_once(monkeypatch):
    said = []
    requests, client = grist(monkeypatch, said)
    monkeypatch.delenv("GRIST_API_KEY")
    for _ in range(2):
        assert client.share("Q?", result(), PROVIDER, "icl") is None
    assert not requests
    assert len(said) == 2 and "GRIST_API_KEY is not set" in said[1]
    with pytest.raises(FeedbackError, match="GRIST_API_KEY"):
        client.rate("7", "correct")
