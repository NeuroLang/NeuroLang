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
import re
import sqlite3

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
    memory REAL              -- the private memory of the process, in MB
                             -- (after an input; the most of the session,
                             -- at its end)
);
CREATE INDEX IF NOT EXISTS activity_user ON activity (user, id);
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
               "response", "failed", "seconds", "cpu", "memory")

    def __init__(self, path):
        self.path = str(path)
        with self._connect() as db:
            _wal(db)
            db.executescript(SCHEMA)

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

    def totals(self) -> list:
        """Return, by user, their inputs run, those failed, their seconds
        and CPU seconds (of their inputs: an end's CPU is its process's in
        all), the most memory of their sessions (an end's is the most of its
        session) and their last event."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT user, "
                "SUM(kind = 'submit') AS inputs, "
                "SUM(kind = 'submit' AND failed) AS failed, "
                "ROUND(SUM(CASE WHEN kind = 'submit' THEN seconds END), 1) "
                "AS seconds, "
                "ROUND(SUM(CASE WHEN kind = 'submit' THEN cpu END), 1) "
                "AS cpu, "
                "ROUND(MAX(memory), 1) AS memory, "
                "MAX(created) AS last "
                "FROM activity GROUP BY user ORDER BY last DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def entries(self, user: str) -> list:
        """Return the events of *user*, the first first."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM activity WHERE user = ? ORDER BY id", (user,)
            ).fetchall()
        return [dict(row) for row in rows]
