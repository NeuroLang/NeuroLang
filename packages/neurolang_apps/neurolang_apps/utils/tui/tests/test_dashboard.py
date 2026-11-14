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


# -- The activity page's data (ActivityLog.report, inputs, uptime) ----------

NOON = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)


def _log_at(log, user, when, **fields):
    log.add(user, "submit", text=fields.pop("text", "ans(x) :- p(x)"),
            engine=fields.pop("engine", "neurosynth"), failed=False,
            created=when.isoformat(timespec="microseconds"), **fields)


def test_a_period_starts_at_its_time_not_at_its_day(tmp_path):
    log = ActivityLog(tmp_path / "a.sqlite")
    _log_at(log, "ann", NOON - timedelta(hours=35, minutes=30))  # 00:30
    _log_at(log, "bob", NOON - timedelta(hours=3), cpu=2.0)
    report = log.report(since=NOON - timedelta(days=1), now=NOON)
    assert report["figures"]["inputs"] == 1
    assert [u["user"] for u in report["users"]] == ["bob"]


def test_the_buckets_are_every_hour_or_day_of_the_zone(tmp_path):
    from zoneinfo import ZoneInfo

    log = ActivityLog(tmp_path / "a.sqlite")
    _log_at(log, "ann", NOON - timedelta(days=10))  # the log began then
    _log_at(log, "ann", NOON - timedelta(minutes=30), cpu=1.5)
    hours = log.report(since=NOON - timedelta(days=1), now=NOON)
    assert hours["bucket"] == "hour" and len(hours["series"]) == 25
    assert sum(b["inputs"] for b in hours["series"]) == 1
    assert hours["series"][-2]["cpu"] == 1.5  # 11:30 is in 11:00
    paris = ZoneInfo("Europe/Paris")
    days = log.report(since=NOON - timedelta(days=7), now=NOON, tz=paris)
    assert days["bucket"] == "day" and len(days["series"]) == 8
    assert days["series"][0]["start"].endswith("T00:00:00+01:00")
    assert not days["clipped"]


def test_the_period_starts_when_the_log_began(tmp_path):
    log = ActivityLog(tmp_path / "a.sqlite")
    _log_at(log, "ann", NOON - timedelta(hours=5))
    report = log.report(since=NOON - timedelta(days=7), now=NOON)
    assert report["clipped"] and report["bucket"] == "hour"
    assert len(report["series"]) == 6


def test_the_uptime_counts_from_the_first_run_and_tells_the_crashes(
    tmp_path
):
    log = ActivityLog(tmp_path / "a.sqlite")
    db = sqlite3.connect(log.path)
    stamp = lambda t: t.isoformat(timespec="microseconds")  # noqa: E731
    with db:
        db.executemany(
            "INSERT INTO runs (started, last_seen, stopped) VALUES (?, ?, ?)",
            [(stamp(NOON - timedelta(hours=24)),
              stamp(NOON - timedelta(hours=18)), None),       # crashed
             (stamp(NOON - timedelta(hours=12)),
              stamp(NOON - timedelta(hours=6)),
              stamp(NOON - timedelta(hours=6))),               # stopped
             (stamp(NOON - timedelta(hours=3)),
              stamp(NOON - timedelta(minutes=1)), None)])      # running
    db.close()
    up = log.uptime(since=NOON - timedelta(days=7), now=NOON)
    assert up["since"].startswith("2026-11-13T12:00")  # the first run
    assert up["percent"] == pytest.approx(100 * 15 / 24, abs=0.01)
    hows = [r["how"] for r in up["runs"]]
    assert hows == ["running", "stopped", "crashed"]
    assert up["crashes"] == 1
    assert log.uptime(now=NOON + timedelta(hours=1))["runs"][0]["how"] \
        == "crashed"  # silent for an hour: not running any more
    assert ActivityLog(tmp_path / "b.sqlite").uptime()["percent"] is None


def test_the_inputs_are_sorted_searched_and_cut(tmp_path):
    from neurolang_apps.utils.tui.activity import SNIPPET_CHARS

    log = ActivityLog(tmp_path / "a.sqlite")
    _log_at(log, "ann", NOON - timedelta(hours=3), seconds=9.0,
            text="ans(x) :- slow(x)", response="r" * 1000)
    _log_at(log, "bob", NOON - timedelta(hours=2), seconds=1.0,
            text=".ask which studies?", tokens_in=500, tokens_out=50)
    log.add("bob", "submit", text="ans(x :-", failed=True,
            created=(NOON - timedelta(hours=1)).isoformat())
    latest = log.inputs()
    assert [r["text"] for r in latest] == ["ans(x :-", ".ask which studies?",
                                           "ans(x) :- slow(x)"]
    assert log.inputs(sort="slowest")[0]["text"] == "ans(x) :- slow(x)"
    assert log.inputs(sort="tokens")[0]["user"] == "bob"
    assert [r["text"] for r in log.inputs(failed=True)] == ["ans(x :-"]
    assert [r["text"] for r in log.inputs(search="STUDIES")] == [
        ".ask which studies?"]
    slow = log.inputs(search="slow")[0]
    assert len(slow["snippet"]) == SNIPPET_CHARS and slow["cut"]
    assert len(log.entry(slow["id"])["response"]) == 1000
    assert log.entry(999) == {}
    assert len(log.inputs(limit=1, offset=1)) == 1
    with pytest.raises(ValueError, match="sort"):
        log.inputs(sort="id; DROP TABLE activity")


def test_a_query_of_the_last_day_leaves_out_the_earlier_hours_of_its_day(
    log
):
    # Datasette's queries compared ISO times with SQLite's (a space for
    # the T): the whole first day passed.
    now = datetime.now(timezone.utc)
    early = (now - timedelta(days=1)).replace(hour=0, minute=0, second=1)
    if early >= now - timedelta(days=1):
        pytest.skip("too close to midnight to tell")
    log.add("eve", "submit", text="x", created=early.isoformat())
    assert "eve" not in {r["user"] for r in _query(log, "by_user", "1")}
    assert "eve" in {r["user"] for r in _query(log, "by_user", "2")}
