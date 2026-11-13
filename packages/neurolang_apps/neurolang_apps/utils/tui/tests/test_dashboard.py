"""Tests of the dashboard of the activity log, and of the runs of the
server it shows the uptime of."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from neurolang_apps.utils.tui.activity import ActivityLog
from neurolang_apps.utils.tui.dashboard import QUERIES, metadata


def _at(days_ago: float) -> str:
    when = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return when.isoformat(timespec="microseconds")


@pytest.fixture
def log(tmp_path):
    log = ActivityLog(tmp_path / "activity.sqlite")
    log.add("ann", "start", text="neurosynth", engine="neurosynth")
    log.add("ann", "submit", text="ans(x) :- p(x)", engine="neurosynth",
            response="x", failed=False, seconds=2.0, cpu=1.5, memory=40.0)
    log.add("bob", "submit", text="ans(x :-", engine="neurosynth",
            response="error", failed=True, seconds=0.1, cpu=0.1,
            memory=20.0)
    log.add("ann", "end", text="idle", engine="neurosynth", cpu=9.0,
            memory=60.0)
    log.add("ann", "submit", text=".ask which studies?",
            engine="neurosynth", failed=False, tokens_in=1200,
            tokens_out=300, llm="albert/m", llm_key="server")
    log.add("bob", "submit", text=".ask which regions?",
            engine="neurosynth", failed=False, tokens_in=100, tokens_out=50,
            llm="anthropic/claude", llm_key="page")
    return log


def _query(log, name, days=""):
    db = sqlite3.connect(log.path)
    db.row_factory = sqlite3.Row
    sql = QUERIES[name][1]
    params = {"days": days} if ":days" in sql else {}
    try:
        return [dict(row) for row in db.execute(sql, params)]
    finally:
        db.close()


def test_each_query_runs_over_any_period(log):
    run = log.start_run()
    log.beat(run)
    for name in QUERIES:
        for days in ("", "7"):
            _query(log, name, days)
    by_user = {r["user"]: r for r in _query(log, "by_user")}
    # The CPU of the inputs only; the most memory, an end's too.
    assert (by_user["ann"]["inputs"], by_user["ann"]["cpu_seconds"],
            by_user["ann"]["most_memory_mb"]) == (2, 1.5, 60.0)
    assert by_user["ann"]["tokens_in"] == 1200
    tokens = [(r["user"], r["llm"], r["whose_key"], r["tokens"])
              for r in _query(log, "tokens")]
    assert tokens == [("ann", "albert/m", "server", 1500),
                      ("bob", "anthropic/claude", "page", 150)]
    assert by_user["bob"]["failed"] == 1
    assert [r["reason"] for r in _query(log, "ends")] == ["idle"]
    [hour] = _query(log, "by_hour", "1")
    assert (hour["inputs"], hour["users"]) == (4, 2)


def test_the_uptime_is_the_share_of_the_period_the_runs_cover(log):
    db = sqlite3.connect(log.path)
    with db:
        db.executemany(
            "INSERT INTO runs (started, last_seen, stopped) VALUES (?, ?, ?)",
            [(_at(12), _at(5), _at(5)),  # 5 of the last 10 days
             (_at(2), _at(0), None)])    # 2 more, running
    db.close()
    [uptime] = _query(log, "uptime", "10")
    assert uptime["uptime_percent"] == pytest.approx(70, abs=0.1)
    runs = _query(log, "runs")
    assert [r["how"] for r in runs] == ["running", "stopped"]


def test_a_run_of_the_server_is_logged_to_its_stop(log):
    run = log.start_run()
    log.beat(run, stopped=True)
    [how] = [r["how"] for r in _query(log, "runs")]
    assert how == "stopped"


def test_the_queries_are_those_of_the_activity_log():
    queries = metadata("activity")["databases"]["activity"]["queries"]
    assert set(queries) == set(QUERIES)
    assert all(q["title"] and q["sql"] for q in queries.values())


def test_datasette_serves_the_log_on_its_socket(log):
    pytest.importorskip("datasette")
    pytest.importorskip("datasette_vega")
    from neurolang_apps.utils.tui.dashboard import PREFIX, Dashboard

    board = Dashboard([log.path])
    try:
        status, headers, body = board.get(PREFIX)
        assert status == 200 and b"NeuroLang REPL: activity" in body
        assert dict(headers)["Content-Type"].startswith("text/html")
        status, _, body = board.get(
            f"{PREFIX}activity/by_user.json?days=&_shape=array")
        assert status == 200
        assert {r["user"] for r in json.loads(body)} == {"ann", "bob"}
        # The link of a page's JSON, with base_url twice, as the charts
        # fetch it.
        status, _, body = board.get(
            f"{PREFIX}{PREFIX[1:]}activity/by_user.json?days=&_shape=array")
        assert status == 200 and json.loads(body)
        status, _, body = board.get(f"{PREFIX}-/plugins.json")
        assert "datasette-vega" in body.decode()
    finally:
        board.close()
    assert not board.directory.exists()


def test_a_log_made_before_the_tokens_takes_them(tmp_path):
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    with db:
        db.execute("CREATE TABLE activity (id INTEGER PRIMARY KEY, "
                   "created TEXT NOT NULL, user TEXT NOT NULL, groups TEXT, "
                   "tab TEXT, engine TEXT, kind TEXT NOT NULL, text TEXT, "
                   "response TEXT, failed INTEGER, seconds REAL, cpu REAL, "
                   "memory REAL)")
    db.close()
    log = ActivityLog(path)
    log.add("ann", "submit", text=".ask x", tokens_in=5, tokens_out=1)
    [totals] = log.totals()
    assert (totals["tokens_in"], totals["tokens_out"]) == (5, 1)
