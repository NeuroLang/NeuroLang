"""
The dashboard of ``neurolang-repl-serve --dashboard``: `Datasette
<https://datasette.io/>`_ on the activity log (``--activity-db``) and the
ratings (``--feedback-db``), at ``/admin/logs/``, for the admins only.

Datasette runs in a process of its own, listening on a unix socket in a
directory only this user reads (no port of the machine); the server
passes it the admins' GET requests (:class:`Dashboard`). Its queries
(:data:`QUERIES`) chart, with the plugin datasette-vega, what the users ran
by hour, by user and by engine, the slowest inputs, the failures, why the
sessions ended, and the uptime of the server.
"""

import http.client
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

#: Where the dashboard is, on the server.
PREFIX = "/admin/logs/"

#: What the dashboard imports, and the extra of the workspace that has it.
IMPORTS = "datasette, datasette_vega"
EXTRA = "--extra dashboard"

# The period of a query, :days (empty: all), as the start of its window.
_SINCE = ("created >= COALESCE(datetime('now', '-' || NULLIF(:days, '') || "
          "' days'), '')")

#: The queries of the activity log, by name: their title and SQL.
QUERIES = {
    "by_hour": ("What the users ran, by hour", f"""
SELECT substr(created, 1, 13) || ':00' AS hour,
       SUM(kind = 'submit') AS inputs,
       SUM(kind = 'submit' AND failed) AS failed,
       COUNT(DISTINCT user) AS users,
       ROUND(SUM(CASE WHEN kind = 'submit' THEN cpu END), 1) AS cpu_seconds,
       ROUND(MAX(memory), 1) AS most_memory_mb
FROM activity WHERE {_SINCE}
GROUP BY hour ORDER BY hour"""),
    "by_day": ("What the users ran, by day", f"""
SELECT substr(created, 1, 10) AS day,
       SUM(kind = 'submit') AS inputs,
       SUM(kind = 'submit' AND failed) AS failed,
       COUNT(DISTINCT user) AS users,
       ROUND(SUM(CASE WHEN kind = 'submit' THEN cpu END), 1) AS cpu_seconds,
       ROUND(MAX(memory), 1) AS most_memory_mb
FROM activity WHERE {_SINCE}
GROUP BY day ORDER BY day"""),
    "by_user": ("What each user ran", f"""
SELECT user, groups,
       SUM(kind = 'submit') AS inputs,
       SUM(kind = 'submit' AND failed) AS failed,
       ROUND(SUM(CASE WHEN kind = 'submit' THEN seconds END), 1) AS seconds,
       ROUND(SUM(CASE WHEN kind = 'submit' THEN cpu END), 1) AS cpu_seconds,
       ROUND(MAX(memory), 1) AS most_memory_mb,
       SUM(kind = 'start') AS starts,
       MAX(created) AS last
FROM activity WHERE {_SINCE}
GROUP BY user ORDER BY cpu_seconds DESC"""),
    "by_engine": ("What ran on each engine", f"""
SELECT engine,
       SUM(kind = 'submit') AS inputs,
       SUM(kind = 'submit' AND failed) AS failed,
       COUNT(DISTINCT user) AS users,
       ROUND(SUM(CASE WHEN kind = 'submit' THEN cpu END), 1) AS cpu_seconds,
       ROUND(MAX(memory), 1) AS most_memory_mb
FROM activity WHERE {_SINCE}
GROUP BY engine ORDER BY inputs DESC"""),
    "slowest": ("The slowest inputs", f"""
SELECT created, user, engine, seconds, cpu, memory, failed, text, response
FROM activity WHERE kind = 'submit' AND {_SINCE}
ORDER BY seconds DESC LIMIT 100"""),
    "failures": ("The inputs that failed", f"""
SELECT created, user, engine, text, response
FROM activity WHERE kind = 'submit' AND failed AND {_SINCE}
ORDER BY created DESC LIMIT 500"""),
    "ends": ("Why the sessions ended", f"""
SELECT text AS reason, COUNT(*) AS sessions,
       ROUND(AVG(cpu), 1) AS mean_cpu_seconds,
       ROUND(MAX(memory), 1) AS most_memory_mb
FROM activity WHERE kind = 'end' AND {_SINCE}
GROUP BY reason ORDER BY sessions DESC"""),
    "uptime": ("The uptime of the server", """
WITH span AS (
    SELECT julianday('now') - COALESCE(NULLIF(:days, ''), 30) AS lo,
           julianday('now') AS hi),
runs_in AS (
    SELECT MAX(julianday(started), lo) AS a,
           MIN(julianday(COALESCE(stopped, last_seen)), hi) AS b
    FROM runs, span)
SELECT ROUND(100.0 * SUM(MAX(b - a, 0)) / (SELECT hi - lo FROM span), 2)
           AS uptime_percent,
       COALESCE(NULLIF(:days, ''), 30) AS days
FROM runs_in"""),
    "runs": ("The runs of the server", """
SELECT started, COALESCE(stopped, last_seen) AS until,
       ROUND((julianday(COALESCE(stopped, last_seen)) - julianday(started))
             * 24, 2) AS hours,
       CASE WHEN stopped IS NOT NULL THEN 'stopped'
            WHEN id = (SELECT MAX(id) FROM runs) THEN 'running'
            ELSE 'ended without stopping (killed, crashed?)' END AS how,
       pid
FROM runs ORDER BY id DESC LIMIT 200"""),
}


def _chart(mark: str, x: str, x_type: str, y: str) -> str:
    """The address, in the page of a query, of datasette-vega's chart of
    the column *y* by *x* (of *x_type*, ordinal or temporal) as *mark*."""
    return (f"#g.mark={mark}&g.x_column={x}&g.x_type={x_type}"
            f"&g.y_column={y}&g.y_type=quantitative")


#: The charts of the queries, by name: their label, their days and chart.
CHARTS = {
    "by_hour": [
        ("inputs by hour, last 7 days", "7",
         _chart("line", "hour", "temporal", "inputs")),
        ("CPU by hour, last 7 days", "7",
         _chart("line", "hour", "temporal", "cpu_seconds")),
        ("users by hour, last 7 days", "7",
         _chart("line", "hour", "temporal", "users")),
    ],
    "by_day": [
        ("inputs by day", "", _chart("bar", "day", "temporal", "inputs")),
        ("most memory by day", "",
         _chart("line", "day", "temporal", "most_memory_mb")),
    ],
    "by_user": [
        ("CPU by user", "", _chart("bar", "user", "ordinal", "cpu_seconds")),
        ("inputs by user", "", _chart("bar", "user", "ordinal", "inputs")),
    ],
    "by_engine": [
        ("inputs by engine", "",
         _chart("bar", "engine", "ordinal", "inputs")),
    ],
}


def metadata(activity: str) -> dict:
    """Return the metadata of the dashboard, whose activity log is the
    database *activity* (its name in Datasette)."""
    return {
        "title": "NeuroLang REPL: activity",
        "description": "What the users ran, the resources it took, and the "
        "uptime of the server. Each query takes :days, the days back "
        "(empty: all; the uptime's, 30). Chart a result with its "
        "\"Show charting options\".",
        "databases": {activity: {"queries": {
            name: {"title": title, "sql": sql.strip(), **({
                "description_html": "Charts: " + ", ".join(
                    f'<a href="?days={days}{chart}">{label}</a>'
                    for label, days, chart in CHARTS[name])
            } if name in CHARTS else {})}
            for name, (title, sql) in QUERIES.items()
        }}},
    }


class _UnixConnection(http.client.HTTPConnection):
    """An HTTP connection to the unix socket *path*."""

    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


class Dashboard:
    """
    Datasette on the SQLite databases *paths* (the activity log first), in
    a process started with the Python *python*, on a unix socket of a new
    directory of this user's only; :meth:`get` passes it a request.
    """

    #: How long Datasette may take to start, and to answer, in seconds.
    START_SECONDS = 60.0
    ANSWER_SECONDS = 60.0
    #: The headers of Datasette's answers that the server passes on.
    HEADERS = ("Content-Type", "Location", "Content-Disposition",
               "Cache-Control")

    def __init__(self, paths, python: str = sys.executable):
        self.directory = Path(tempfile.mkdtemp(prefix="neurolang-dashboard-"))
        self.socket = str(self.directory / "datasette.sock")
        paths = [str(Path(p).resolve()) for p in paths]
        settings = self.directory / "metadata.json"
        settings.write_text(json.dumps(metadata(Path(paths[0]).stem)))
        self.process = subprocess.Popen([
            python, "-m", "datasette", "serve", *paths,
            "--uds", self.socket, "--metadata", str(settings),
            "--setting", "base_url", PREFIX,
            "--setting", "sql_time_limit_ms", "20000",
        ], stdout=subprocess.DEVNULL)  # a line a request; its errors stay
        deadline = time.monotonic() + self.START_SECONDS
        while not Path(self.socket).exists():
            if self.process.poll() is not None or \
                    time.monotonic() > deadline:
                self.close()
                raise RuntimeError("The dashboard (Datasette) did not start: "
                                   "see the server's log")
            time.sleep(0.2)

    def get(self, target: str):
        """Return Datasette's answer to a GET of *target* (its path, of
        PREFIX, and its query): its status, the headers passed on and its
        body."""
        # Datasette (0.65) doubles its base_url in the link of a page's
        # JSON, which the charts fetch: one is the page's.
        doubled = PREFIX + PREFIX[1:]
        if target.startswith(doubled):
            target = PREFIX + target[len(doubled):]
        if self.process.poll() is not None:
            return 502, [("Content-Type", "text/plain")], \
                b"The dashboard (Datasette) ended: see the server's log"
        connection = _UnixConnection(self.socket, self.ANSWER_SECONDS)
        try:
            connection.request("GET", target, headers={"Host": "localhost"})
            response = connection.getresponse()
            body = response.read()
            headers = [(name.title(), value)
                       for name, value in response.getheaders()
                       if name.title() in self.HEADERS]
            return response.status, headers, body
        finally:
            connection.close()

    def close(self) -> None:
        """End Datasette and remove its directory."""
        self.process.terminate()
        try:
            self.process.wait(10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        shutil.rmtree(self.directory, ignore_errors=True)
