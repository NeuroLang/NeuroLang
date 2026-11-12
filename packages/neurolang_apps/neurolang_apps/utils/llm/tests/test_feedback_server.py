"""Tests of the feedback server, run in a thread on an ephemeral port."""

import contextlib
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
    assert bank == {"assistant_examples": [
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


def _rate_answers(path, writer, count):
    """Add *count* answers to the store at *path*, each rated (in a process
    of its own)."""
    store = FeedbackStore(path)
    for number in range(count):
        answer = store.add_answer({"question": f"q{writer}-{number}"})
        store.add_feedback(answer, {"verdict": "correct"})


def test_concurrent_writers_lose_nothing(tmp_path):
    import multiprocessing

    path = tmp_path / "feedback.sqlite"
    FeedbackStore(path)
    context = multiprocessing.get_context("spawn")
    writers = [context.Process(target=_rate_answers, args=(path, w, 25))
               for w in range(4)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(120)
        assert writer.exitcode == 0
    answers = FeedbackStore(path).answers()
    assert len(answers) == 100
    assert len({answer["question"] for answer in answers}) == 100


def _old_database(path):
    """A database of the tables as they were before the groups."""
    import sqlite3

    db = sqlite3.connect(path)
    with db:
        db.executescript(
            "CREATE TABLE answers (id TEXT PRIMARY KEY, created TEXT,"
            " client_version TEXT, engine TEXT, provider TEXT, model TEXT,"
            " strategy TEXT, language TEXT, question TEXT NOT NULL,"
            " query TEXT, explanation TEXT, steps TEXT, rows TEXT,"
            " valid INTEGER, error TEXT, seconds REAL, calls INTEGER,"
            " tokens INTEGER, user TEXT);"
            "CREATE TABLE feedback (id TEXT PRIMARY KEY, answer_id TEXT NOT"
            " NULL REFERENCES answers(id), created TEXT, user TEXT, verdict"
            " TEXT NOT NULL, comment TEXT, corrected_query TEXT,"
            " corrected_query_checked INTEGER, corrected_rows TEXT);"
            "INSERT INTO answers (id, question) VALUES ('a1', 'old one');"
        )
    db.close()


def test_a_database_made_before_the_groups_is_kept(tmp_path):
    path = tmp_path / "old.sqlite"
    _old_database(path)
    store = FeedbackStore(path)
    store.add_answer({"question": "new one", "user": "ann", "groups": "lab"})
    answers = {a["question"]: a for a in store.answers()}
    assert answers["old one"]["groups"] is None
    assert answers["new one"]["groups"] == "lab"


def test_a_rating_of_the_page_is_stored_signed(tmp_path):
    from ..feedback import StoreFeedbackClient, query_record

    path = tmp_path / "feedback.sqlite"
    client = StoreFeedbackClient(path, "ann", "admin,lab")
    client.share_record(
        query_record("ans(x) :- p(x)", "1 row", "", "neurosynth"),
        "bad", "wrong rows",
    )
    [answer] = FeedbackStore(path).answers()
    assert (answer["user"], answer["groups"], answer["status"]) == (
        "ann", "admin,lab", "incorrect")
    [rating] = answer["feedback"]
    assert (rating["user"], rating["comment"]) == ("ann", "wrong rows")


def _open_at(path, when):
    """Open the store *path* at the time *when* (in a process of its own)."""
    import time

    while time.time() < when:
        pass
    FeedbackStore(path)


@pytest.mark.parametrize("old", [True, False])
def test_processes_opening_a_database_at_once_all_open_it(tmp_path, old):
    import multiprocessing
    import time

    context = multiprocessing.get_context("spawn")
    for round in range(4):
        path = tmp_path / f"feedback-{round}.sqlite"
        if old:
            _old_database(path)
        when = time.time() + 2  # once all are spawned
        openers = [context.Process(target=_open_at, args=(path, when))
                   for _ in range(6)]
        for opener in openers:
            opener.start()
        for opener in openers:
            opener.join(60)
        assert [opener.exitcode for opener in openers] == [0] * 6
        store = FeedbackStore(path)
        store.add_answer({"question": "q", "groups": "lab"})


def test_a_read_only_database_is_read_and_exported(tmp_path, capsys):
    import sqlite3

    path = tmp_path / "feedback.sqlite"
    _old_database(path)  # made before the groups: not as they are now
    with contextlib.closing(sqlite3.connect(path)) as db, db:
        db.execute("INSERT INTO feedback (id, answer_id, verdict) VALUES "
                   "('f1', 'a1', 'correct')")
    path.chmod(0o444)
    try:
        assert [a["question"] for a in FeedbackStore(path).answers()] == [
            "old one"]
        main(["export", "--db", str(path)])
        assert "old one" in capsys.readouterr().out
    finally:
        path.chmod(0o644)


def test_the_store_lists_its_engines(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite")
    for engine in ("neurosynth", "neurostore", "neurosynth"):
        store.add_answer({**ANSWER, "engine": engine})
    store.add_answer({"question": "no engine"})
    assert store.engines() == ["neurostore", "neurosynth"]
