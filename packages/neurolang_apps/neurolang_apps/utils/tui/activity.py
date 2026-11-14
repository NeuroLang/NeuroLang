"""
What the users of ``neurolang-repl-serve --local --users FILE
--activity-db DB`` do: the starts and ends of their sessions, and each
input they run, with its result as text, how long it took, the CPU time
and the memory of its session's process, in the SQLite database *DB*.

The admins' console shows the totals of each user and gives the log of a
user (:class:`ActivityLog`). The page of a user tells them their inputs are
logged.
"""

import contextlib
import html
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone

from neurolang_apps.utils.llm.feedback_server import _now, _wal

SCHEMA = """
CREATE TABLE IF NOT EXISTS activity (
    id INTEGER PRIMARY KEY,
    created TEXT NOT NULL,
    user TEXT NOT NULL,
    groups TEXT,
    tab TEXT,
    engine TEXT,
    kind TEXT NOT NULL,      -- start, submit or end
    text TEXT,               -- the input; the engine; why it ended
    response TEXT,           -- the result, as text
    failed INTEGER,
    seconds REAL,            -- how long it took
    cpu REAL,                -- the CPU seconds of its session's process
    memory REAL,             -- the private memory of the process, in MB
                             -- (after an input; the most of the session,
                             -- at its end)
    tokens_in INTEGER,       -- the tokens the assistant sent and received
    tokens_out INTEGER,      -- answering the input, if it did,
    llm TEXT,                -- with the provider/model,
    llm_key TEXT             -- and whose key: page (theirs), server
);
CREATE INDEX IF NOT EXISTS activity_user ON activity (user, id);
CREATE INDEX IF NOT EXISTS activity_created ON activity (created);
-- Each run of the server: its start, the last time it was alive (every
-- minute) and its end, if it stopped (a run without one ended otherwise:
-- killed, crashed, the machine stopped).
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    started TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    stopped TEXT,
    pid INTEGER
);
"""

#: The most characters of a result kept.
RESPONSE_CHARS = 20000

#: What the page of a user is told when their inputs are logged.
LOG_NOTE = (
    "This server logs what you run and its results, with your user name, "
    "the time and the resources it took, for its admins."
)


#: The most rows of a table of a result kept.
RESPONSE_ROWS = 50
#: The characters of a result an input list shows (the rest by its id).
SNIPPET_CHARS = 300


def _stamp(when: datetime) -> str:
    """*when* as the log writes it (UTC, ISO, microseconds), so that the
    stamps compare as text."""
    return when.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def _floor(when: datetime, tz, hourly: bool) -> str:
    """The start of the hour, or day, of *when* in *tz*, as a stamp of
    that zone (what the page shows)."""
    local = when.astimezone(tz)
    local = local.replace(minute=0, second=0, microsecond=0)
    if not hourly:
        local = local.replace(hour=0)
    return local.isoformat()


def _buckets(start: datetime, end: datetime, tz, hourly: bool) -> list:
    """Every hour, or day, of *tz* from that of *start* to that of *end*,
    empty."""
    buckets, current = [], _floor(start, tz, hourly)
    last = _floor(end, tz, hourly)
    while True:
        buckets.append({"start": current, "inputs": 0, "failed": 0,
                        "cpu": 0.0, "tokens": 0, "users": set()})
        if current >= last:
            return buckets
        local = datetime.fromisoformat(current)
        # the next one, in the zone's own clock (a day of 23 or 25 hours)
        step = (local + timedelta(hours=1, minutes=30) if hourly
                else local + timedelta(days=1, hours=12))
        current = _floor(step.astimezone(timezone.utc), tz, hourly)


def response_text(reply: dict) -> str:
    """Return the *reply* of the REPL as text: its output, then the first
    RESPONSE_ROWS rows of each table, tab-separated, cut at
    RESPONSE_CHARS."""
    text = html.unescape(re.sub(r"<[^>]*>", "", reply.get("html") or ""))
    for table in reply.get("tables") or ():
        rows = [table.get("columns", [])] + table.get("rows", [])[
            :RESPONSE_ROWS]
        text += "\n" + "\n".join("\t".join(map(str, row)) for row in rows)
        if table.get("total", 0) > RESPONSE_ROWS:
            text += f"\n[{table['total']} rows]"
    if len(text) > RESPONSE_CHARS:
        text = text[:RESPONSE_CHARS] + "\n[…]"
    return text


class ActivityLog:
    """
    The activity of the users, in the SQLite database *path*, written by
    the threads of the server at once: each write is a transaction of a
    connection of its own, in WAL mode, as in
    :class:`~neurolang_apps.utils.llm.feedback_server.FeedbackStore`.
    """

    COLUMNS = ("created", "user", "groups", "tab", "engine", "kind", "text",
               "response", "failed", "seconds", "cpu", "memory",
               "tokens_in", "tokens_out", "llm", "llm_key")
    #: The columns a log made before them lacks.
    ADDED = {"tokens_in": "INTEGER", "tokens_out": "INTEGER", "llm": "TEXT",
             "llm_key": "TEXT"}

    def __init__(self, path):
        self.path = str(path)
        with self._connect() as db:
            _wal(db)
            db.executescript(SCHEMA)
            columns = {row["name"] for row in db.execute(
                "PRAGMA table_info(activity)")}
            for name, kind in self.ADDED.items():
                if name not in columns:
                    try:
                        db.execute(
                            f"ALTER TABLE activity ADD COLUMN {name} {kind}")
                    except sqlite3.OperationalError as error:
                        if "duplicate column" not in str(error):
                            raise  # not another process adding it at once

    @contextlib.contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, user: str, kind: str, **fields) -> None:
        """Log the event *kind* of *user*, with *fields* (of COLUMNS)."""
        row = {"created": _now(), "user": user, "kind": kind}
        row.update((k, v) for k, v in fields.items() if k in self.COLUMNS)
        with self._connect() as db:
            db.execute(
                f"INSERT INTO activity ({', '.join(row)}) "
                f"VALUES ({', '.join('?' * len(row))})", tuple(row.values()),
            )

    @staticmethod
    def _where(since=None, user=None, engine=None) -> tuple:
        """The WHERE clause of the events from *since* (a datetime), of
        *user* and on *engine*, those given, and its parameters."""
        clauses, params = ["1"], []
        if since is not None:
            clauses.append("created >= ?")
            params.append(_stamp(since))
        if user:
            clauses.append("user = ?")
            params.append(user)
        if engine:
            clauses.append("engine = ?")
            params.append(engine)
        return " AND ".join(clauses), params

    def totals(self, since=None, user=None, engine=None) -> list:
        """Return, by user, their inputs run, those failed, their seconds
        and CPU seconds (of their inputs: an end's CPU is its process's in
        all), the most memory of their sessions (an end's is the most of its
        session), their tokens and their last event; of the events from
        *since*, of *user* and on *engine*, if given."""
        where, params = self._where(since, user, engine)
        with self._connect() as db:
            rows = db.execute(
                "SELECT user, MAX(groups) AS groups, "
                "SUM(kind = 'submit') AS inputs, "
                "SUM(kind = 'submit' AND failed) AS failed, "
                "ROUND(SUM(CASE WHEN kind = 'submit' THEN seconds END), 1) "
                "AS seconds, "
                "ROUND(SUM(CASE WHEN kind = 'submit' THEN cpu END), 1) "
                "AS cpu, "
                "ROUND(MAX(memory), 1) AS memory, "
                "SUM(tokens_in) AS tokens_in, SUM(tokens_out) AS tokens_out, "
                "MAX(created) AS last "
                f"FROM activity WHERE {where} GROUP BY user "
                "ORDER BY last DESC", params,
            ).fetchall()
        return [dict(row) for row in rows]

    #: How an input list may be sorted: the column, the first first.
    SORTS = {"latest": "created DESC", "slowest": "seconds DESC",
             "cpu": "cpu DESC", "tokens":
             "COALESCE(tokens_in, 0) + COALESCE(tokens_out, 0) DESC"}

    def inputs(self, since=None, user=None, engine=None, failed=False,
               search="", sort="latest", limit=50, offset=0) -> list:
        """Return the inputs (from *since*, of *user*, on *engine*, those
        that failed only if *failed*, whose text holds *search*), sorted as
        SORTS says, *limit* from *offset*; each with the start of its
        result (``snippet``), the whole being :meth:`entry`'s."""
        if sort not in self.SORTS:
            raise ValueError(f"sort is one of {', '.join(self.SORTS)}")
        where, params = self._where(since, user, engine)
        where += " AND kind = 'submit'"
        if failed:
            where += " AND failed"
        if search:
            where += " AND instr(lower(text), lower(?)) > 0"
            params.append(search)
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, created, user, engine, text, failed, seconds, "
                "cpu, memory, tokens_in, tokens_out, llm, llm_key, "
                f"substr(response, 1, {SNIPPET_CHARS}) AS snippet, "
                f"length(response) > {SNIPPET_CHARS} AS cut "
                f"FROM activity WHERE {where} "
                f"ORDER BY {self.SORTS[sort]}, id DESC LIMIT ? OFFSET ?",
                [*params, int(limit), int(offset)],
            ).fetchall()
        return [dict(row) for row in rows]

    def entry(self, ident: int) -> dict:
        """Return the event *ident*, whole, or {} if there is none."""
        with self._connect() as db:
            row = db.execute("SELECT * FROM activity WHERE id = ?",
                             (int(ident),)).fetchone()
        return dict(row) if row else {}

    def report(self, since=None, user=None, engine=None, tz=None,
               now=None) -> dict:
        """
        Return what the dashboard shows of the events from *since* (a
        datetime, aware; all if None), or from when the log began if later
        (``since``, ``clipped``), of *user* and on *engine*: the
        ``figures`` of the period, its ``series`` by hour (2 days at most)
        or by day in the time zone *tz* (a :class:`zoneinfo.ZoneInfo`,
        UTC by default), every bucket of the period there, empty or not,
        why its sessions ``ends``, the ``users``, ``names`` and
        ``engines`` of the whole log, and the ``uptime`` of the server.
        """
        tz = tz or timezone.utc
        now = now or datetime.now(timezone.utc)
        where, params = self._where(since, user, engine)
        with self._connect() as db:
            events = db.execute(
                "SELECT created, kind, user, failed, cpu, memory, "
                "tokens_in, tokens_out, llm_key, text FROM activity "
                f"WHERE {where} ORDER BY id", params).fetchall()
            names = [r[0] for r in db.execute(
                "SELECT DISTINCT user FROM activity ORDER BY user")]
            engines = [r[0] for r in db.execute(
                "SELECT DISTINCT engine FROM activity WHERE engine != '' "
                "ORDER BY engine")]
            # the log began with its first event or run: nothing is known
            # before, which is not a time the server was down
            first = db.execute(
                "SELECT MIN(t) FROM (SELECT MIN(created) AS t FROM activity "
                "UNION ALL SELECT MIN(started) FROM runs)").fetchone()[0]
        began = _parse(first) if first else now
        start = max(since, began) if since else began
        hourly = now - start <= timedelta(days=2)
        series = _buckets(start, now, tz, hourly)
        index = {b["start"]: b for b in series}
        figures = {"inputs": 0, "failed": 0, "cpu": 0.0, "tokens_in": 0,
                   "tokens_out": 0, "tokens_server": 0, "memory": None,
                   "sessions": 0, "users": 0}
        users, ends = set(), {}
        for e in events:
            if e["kind"] == "start":
                figures["sessions"] += 1
            if e["kind"] == "end":
                ends[e["text"]] = ends.get(e["text"], 0) + 1
            if e["memory"] is not None:
                figures["memory"] = max(figures["memory"] or 0, e["memory"])
            if e["kind"] != "submit":
                continue
            tokens = (e["tokens_in"] or 0) + (e["tokens_out"] or 0)
            figures["inputs"] += 1
            figures["failed"] += bool(e["failed"])
            figures["cpu"] += e["cpu"] or 0
            figures["tokens_in"] += e["tokens_in"] or 0
            figures["tokens_out"] += e["tokens_out"] or 0
            if e["llm_key"] == "server":
                figures["tokens_server"] += tokens
            users.add(e["user"])
            bucket = index.get(_floor(_parse(e["created"]), tz, hourly))
            if bucket is not None:
                bucket["inputs"] += 1
                bucket["failed"] += bool(e["failed"])
                bucket["cpu"] += e["cpu"] or 0
                bucket["tokens"] += tokens
                bucket["users"].add(e["user"])
        figures["users"] = len(users)
        figures["cpu"] = round(figures["cpu"], 1)
        if figures["memory"] is not None:
            figures["memory"] = round(figures["memory"], 1)
        for bucket in series:
            bucket["users"] = len(bucket["users"])
            bucket["cpu"] = round(bucket["cpu"], 1)
        return {
            "since": _stamp(start), "now": _stamp(now),
            "clipped": since is None or start > since,
            "bucket": "hour" if hourly else "day", "figures": figures,
            "series": series,
            "ends": sorted(({"reason": k, "sessions": v}
                            for k, v in ends.items()),
                           key=lambda e: -e["sessions"]),
            "users": self.totals(since, user, engine),
            "names": names, "engines": engines,
            "uptime": self.uptime(since, now),
        }

    #: How long a run's last beat may be old and the run still running.
    RUNNING = timedelta(minutes=3)

    def uptime(self, since=None, now=None) -> dict:
        """
        Return the share of the period from *since* (or the first run) to
        *now* that the runs of the server cover, in percent, and from when
        it counts (the first run, if later than *since*: the log knows
        nothing before); the ``runs`` that overlap it, the latest first,
        each with its ``start``, ``end`` and how it ended (``stopped``, or
        ``running`` if it is the last and alive, else killed or crashed);
        and the ``crashes`` among them.
        """
        now = now or datetime.now(timezone.utc)
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, started, last_seen, stopped FROM runs "
                "ORDER BY id").fetchall()
        if not rows:
            return {"percent": None, "since": None, "runs": [],
                    "crashes": 0}
        lo = max(since or _parse(rows[0]["started"]),
                 _parse(rows[0]["started"]))
        covered, runs, crashes = timedelta(0), [], 0
        for i, row in enumerate(rows):
            start = _parse(row["started"])
            last = i == len(rows) - 1
            if row["stopped"]:
                end, how = _parse(row["stopped"]), "stopped"
            elif last and now - _parse(row["last_seen"]) <= self.RUNNING:
                end, how = now, "running"
            else:
                end, how = _parse(row["last_seen"]), "crashed"
            if end <= lo:
                continue
            covered += min(end, now) - max(start, lo)
            crashes += how == "crashed"
            runs.append({"start": _stamp(start), "end": _stamp(end),
                         "how": how})
        span = (now - lo).total_seconds()
        percent = 100.0 if span <= 0 else round(
            100 * covered.total_seconds() / span, 2)
        return {"percent": min(percent, 100.0), "since": _stamp(lo),
                "runs": runs[::-1], "crashes": crashes}

    def start_run(self) -> int:
        """Log a start of the server, and return the number of its run."""
        now = _now()
        with self._connect() as db:
            return db.execute(
                "INSERT INTO runs (started, last_seen, pid) VALUES (?, ?, ?)",
                (now, now, os.getpid()),
            ).lastrowid

    def beat(self, run: int, stopped: bool = False) -> None:
        """Log that the run *run* of the server is alive now, or, if
        *stopped*, that it stops now."""
        now = _now()
        with self._connect() as db:
            db.execute(
                "UPDATE runs SET last_seen = ?, stopped = "
                "CASE WHEN ? THEN ? ELSE stopped END WHERE id = ?",
                (now, stopped, now, run),
            )

    def entries(self, user: str) -> list:
        """Return the events of *user*, the first first."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM activity WHERE user = ? ORDER BY id", (user,)
            ).fetchall()
        return [dict(row) for row in rows]
