"""Tests of the feedback server, run in a thread on an ephemeral port."""

import json
import threading
import urllib.error
import urllib.request

import pytest
import yaml

from ..feedback_server import FeedbackStore, examples_yaml, main, make_server

TOKEN = "s3cret"
ANSWER = {
    "engine": "neurosynth", "provider": "fake", "model": "m",
    "strategy": "react", "language": "datalog",
    "question": "Which studies mention pain?",
    "query": "ans(s) :- mentions(s, 'pain')", "valid": True,
    "steps": [{"thought": "t", "action": "run_query",
               "arguments": {"query": "q"}, "observation": "1 rows"}],
    "rows": "s\n1", "seconds": 1.5, "calls": 2, "tokens": 30,
}


@pytest.fixture
def server(tmp_path):
    server = make_server(
        tmp_path / "feedback.sqlite", port=0, admin_token=TOKEN,
        max_bytes=10_000, per_minute=1000,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def request(server, path, data=None, raw=None, headers=None):
    """Return the status and the body of a request to *server*."""
    body = raw if raw is not None else (
        None if data is None else json.dumps(data).encode()
    )
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}", data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def post_answer(server, **fields):
    status, body = request(server, "/api/answers", {**ANSWER, **fields})
    assert status == 201
    return json.loads(body)["id"]


def test_answer_feedback_and_list(server):
    answer = post_answer(server)
    other = post_answer(server, engine="neurostore", question="Fear?")
    status, _ = request(server, f"/api/answers/{answer}/feedback", {
        "verdict": "correct", "user": "ann", "comment": "good",
    })
    assert status == 201
    listed = json.loads(request(server, "/api/answers")[1])
    assert listed["total"] == 2
    assert [a["id"] for a in listed["answers"]] == [other, answer]
    rated = json.loads(request(server, "/api/answers?status=correct")[1])
    assert [a["id"] for a in rated["answers"]] == [answer]
    record = rated["answers"][0]
    assert record["steps"] == ANSWER["steps"] and record["valid"] is True
    assert record["feedback"][0]["verdict"] == "correct"
    unrated = json.loads(request(server, "/api/answers?status=unrated")[1])
    assert [a["id"] for a in unrated["answers"]] == [other]
    engine = json.loads(request(server, "/api/answers?engine=neurostore")[1])
    assert [a["id"] for a in engine["answers"]] == [other]
    page = json.loads(request(server, "/api/answers?limit=1&offset=1")[1])
    assert len(page["answers"]) == 1 and page["total"] == 2


def test_export_needs_the_token(server):
    answer = post_answer(server)
    request(server, f"/api/answers/{answer}/feedback", {
        "verdict": "incorrect", "corrected_query": "ans(s) :- study(s)",
        "corrected_query_checked": True,
    })
    assert request(server, "/api/export.jsonl")[0] == 401
    status, body = request(
        server, "/api/export.jsonl",
        headers={"Authorization": "Bearer wrong"},
    )
    assert status == 401
    status, body = request(
        server, "/api/export.jsonl",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert status == 200
    (record,) = [json.loads(line) for line in body.splitlines()]
    assert record["training"] == {
        "question": ANSWER["question"], "query": "ans(s) :- study(s)",
        "language": "datalog", "engine": "neurosynth", "source": "corrected",
    }


def test_export_disabled_without_a_token(tmp_path):
    server = make_server(tmp_path / "db.sqlite", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert request(server, "/api/export.jsonl")[0] == 403
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("path, body, status", [
    ("/api/answers", b"{not json", 400),
    ("/api/answers", b"[]", 400),
    ("/api/answers", json.dumps({"query": "q"}).encode(), 400),
    ("/api/answers", json.dumps({**ANSWER, "api_key": "k"}).encode(), 400),
    ("/api/answers", json.dumps({**ANSWER, "calls": "2"}).encode(), 400),
    ("/api/answers", json.dumps({**ANSWER, "valid": 1}).encode(), 400),
    ("/api/answers", json.dumps(
        {**ANSWER, "rows": "x" * 20_000}).encode(), 413),
    (f"/api/answers/{'0' * 32}/feedback", b'{"verdict": "correct"}', 404),
    ("/api/nothing", b"{}", 404),
])
def test_bad_requests(server, path, body, status):
    assert request(server, path, raw=body)[0] == status


def test_bad_verdict(server):
    answer = post_answer(server)
    status, body = request(
        server, f"/api/answers/{answer}/feedback", {"verdict": "great"}
    )
    assert status == 400 and "verdict" in body


def test_rate_limit(tmp_path):
    server = make_server(tmp_path / "db.sqlite", port=0, per_minute=2)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        statuses = [
            request(server, "/api/answers", ANSWER)[0] for _ in range(3)
        ]
        assert statuses == [201, 201, 429]
    finally:
        server.shutdown()
        server.server_close()


def test_review_page_and_cors(server):
    status, page = request(server, "/")
    assert status == 200 and "api/answers" in page and "<script>" in page
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/api/answers",
        method="OPTIONS",
    )
    with urllib.request.urlopen(req) as response:
        assert response.status == 204
        assert response.headers["Access-Control-Allow-Origin"] == "*"
        assert "POST" in response.headers["Access-Control-Allow-Methods"]


def test_examples_export(tmp_path, capsys):
    db = tmp_path / "db.sqlite"
    store = FeedbackStore(db)
    approved = store.add_answer({**ANSWER, "question": "Pain?"})
    store.add_feedback(approved, {"verdict": "correct"})
    store.add_feedback(approved, {"verdict": "correct"})
    store.add_feedback(approved, {"verdict": "incorrect"})
    corrected = store.add_answer({**ANSWER, "question": "Fear?"})
    store.add_feedback(corrected, {
        "verdict": "incorrect", "corrected_query_checked": True,
        "corrected_query": "a(s) :- mentions(s, 'fear')\nans(s) :- a(s)",
    })
    unchecked = store.add_answer({**ANSWER, "question": "Memory?"})
    store.add_feedback(unchecked, {
        "verdict": "incorrect", "corrected_query": "ans(s) :- study(s)",
    })
    store.add_answer({**ANSWER, "question": "Unrated?"})
    bank = yaml.safe_load(examples_yaml(store))
    assert bank == {"examples": [
        {"question": "Fear?",
         "datalog": "a(s) :- mentions(s, 'fear')\nans(s) :- a(s)"},
        {"question": "Pain?", "datalog": ANSWER["query"]},
    ]}
    output = tmp_path / "bank.yaml"
    main(["export", "--db", str(db), "--format", "examples",
          "-o", str(output)])
    assert yaml.safe_load(output.read_text()) == bank
    main(["export", "--db", str(db)])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert all("steps" in json.loads(line) for line in lines)
