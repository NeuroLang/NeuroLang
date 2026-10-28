"""
A server collecting the answers of the query assistant and their ratings.

The assistants that share their answers (``feedback`` in the LLM
configuration, see :mod:`.feedback`) post them here; people rate them
correct, incorrect or unsure and correct their query, from the REPL
(``.rate``, ``.refine``) or the review page the server shows at ``/``.
The answers, their ratings and the corrected queries are exported to
train models and to grow the banks of examples::

    neurolang-feedback-server --db feedback.sqlite --port 8765
    neurolang-feedback-server export --db feedback.sqlite --format examples

It needs the standard library only: ``http.server`` and ``sqlite3``.
Anyone reaching the server can post answers and ratings; the export
needs the administrator's token.

Endpoints:

``POST /api/answers``
    An answer (see :data:`ANSWER_FIELDS`); returns ``{"id": ...}``.
``POST /api/answers/ID/feedback``
    A rating of answer ID (see :data:`FEEDBACK_FIELDS`).
``GET /api/answers?engine=&status=&limit=&offset=``
    The answers, latest first, with their ratings; *status* is
    ``unrated`` or the majority verdict.
``GET /api/export.jsonl``
    Every answer with its ratings and its ``training`` pair, one JSON
    per line, given the header ``Authorization: Bearer TOKEN``.
"""

import argparse
import hmac
import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VERDICTS = ("correct", "incorrect", "unsure")
STATUSES = ("unrated",) + VERDICTS
ADMIN_TOKEN_ENV = "NEUROLANG_FEEDBACK_ADMIN_TOKEN"
MAX_BYTES = 1_000_000
POSTS_PER_MINUTE = 60
PAGE_SIZE = 20
# The fields of an answer and of a rating, with their JSON types.
ANSWER_FIELDS = {
    "client_version": str, "engine": str, "provider": str, "model": str,
    "strategy": str, "language": str, "question": str, "query": str,
    "explanation": str, "steps": list, "rows": str, "valid": bool,
    "error": str, "seconds": float, "calls": int, "tokens": int,
    "user": str,
}
FEEDBACK_FIELDS = {
    "user": str, "verdict": str, "comment": str, "corrected_query": str,
    "corrected_query_checked": bool, "corrected_rows": str,
}
SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id TEXT PRIMARY KEY, created TEXT, client_version TEXT, engine TEXT,
    provider TEXT, model TEXT, strategy TEXT, language TEXT,
    question TEXT NOT NULL, query TEXT, explanation TEXT, steps TEXT,
    rows TEXT, valid INTEGER, error TEXT, seconds REAL, calls INTEGER,
    tokens INTEGER, user TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY, answer_id TEXT NOT NULL REFERENCES answers(id),
    created TEXT, user TEXT, verdict TEXT NOT NULL, comment TEXT,
    corrected_query TEXT, corrected_query_checked INTEGER,
    corrected_rows TEXT
);
CREATE INDEX IF NOT EXISTS feedback_answer ON feedback(answer_id);
"""
# The answers with their status: unrated, or the majority verdict.
ANSWERS = """
SELECT * FROM (
    SELECT a.*, CASE
        WHEN v.n IS NULL THEN 'unrated'
        WHEN v.c > v.i THEN 'correct'
        WHEN v.i > v.c THEN 'incorrect'
        ELSE 'unsure' END AS status
    FROM answers a LEFT JOIN (
        SELECT answer_id, COUNT(*) AS n, SUM(verdict = 'correct') AS c,
               SUM(verdict = 'incorrect') AS i
        FROM feedback GROUP BY answer_id
    ) v ON v.answer_id = a.id
) WHERE (:engine IS NULL OR engine = :engine)
    AND (:status IS NULL OR status = :status)
ORDER BY created DESC
"""
_FEEDBACK_PATH = re.compile(r"^/api/answers/([0-9a-f]{32})/feedback$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def validate(data, fields: dict, required: tuple) -> dict:
    """Return the JSON object *data* checked against *fields*."""
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(unknown)}")
    for name in required:
        if not data.get(name):
            raise ValueError(f"missing field: {name}")
    for name, value in data.items():
        kind = fields[name]
        ok = (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            if kind is float else
            isinstance(value, kind) and (
                kind is bool or not isinstance(value, bool)
            )
        )
        if value is not None and not ok:
            raise ValueError(f"{name} must be a {kind.__name__}")
    return data


class FeedbackStore:
    """The answers and their ratings, in the SQLite database *path*."""

    def __init__(self, path):
        self.path = str(path)
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def add_answer(self, data: dict) -> str:
        """Store the answer *data* and return its id."""
        data = validate(data, ANSWER_FIELDS, ("question",))
        row = {name: data.get(name) for name in ANSWER_FIELDS}
        row["steps"] = json.dumps(row["steps"] or [])
        row.update(id=uuid.uuid4().hex, created=_now())
        with self._connect() as db:
            db.execute(
                f"INSERT INTO answers ({', '.join(row)}) "
                f"VALUES ({', '.join(':' + name for name in row)})", row,
            )
        return row["id"]

    def add_feedback(self, answer_id: str, data: dict) -> str:
        """Store the rating *data* of the answer *answer_id*: its id."""
        data = validate(data, FEEDBACK_FIELDS, ("verdict",))
        if data["verdict"] not in VERDICTS:
            raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
        row = {name: data.get(name) for name in FEEDBACK_FIELDS}
        row.update(id=uuid.uuid4().hex, answer_id=answer_id, created=_now())
        with self._connect() as db:
            if not db.execute(
                "SELECT 1 FROM answers WHERE id = ?", (answer_id,)
            ).fetchone():
                raise KeyError(answer_id)
            db.execute(
                f"INSERT INTO feedback ({', '.join(row)}) "
                f"VALUES ({', '.join(':' + name for name in row)})", row,
            )
        return row["id"]

    def answers(self, engine=None, status=None, limit=None, offset=0):
        """Return the answers, latest first, with their ratings."""
        query = ANSWERS + (" LIMIT :limit OFFSET :offset" if limit else "")
        with self._connect() as db:
            rows = db.execute(query, {
                "engine": engine or None, "status": status or None,
                "limit": limit, "offset": offset,
            }).fetchall()
            ratings = defaultdict(list)
            ids = [row["id"] for row in rows]
            for rating in db.execute(
                "SELECT * FROM feedback WHERE answer_id IN "
                f"({', '.join('?' * len(ids))}) ORDER BY created", ids,
            ):
                rating = dict(rating)
                rating["corrected_query_checked"] = bool(
                    rating["corrected_query_checked"]
                )
                ratings[rating.pop("answer_id")].append(rating)
        records = []
        for row in rows:
            record = dict(row)
            record["steps"] = json.loads(record["steps"] or "[]")
            record["valid"] = bool(record["valid"])
            record["feedback"] = ratings[record["id"]]
            records.append(record)
        return records

    def count(self, engine=None, status=None) -> int:
        """Return the number of answers of *engine* with *status*."""
        with self._connect() as db:
            return db.execute(
                f"SELECT COUNT(*) FROM ({ANSWERS})",
                {"engine": engine or None, "status": status or None},
            ).fetchone()[0]


def training_pair(record: dict):
    """
    Return the question of *record* with its best query, or None.

    The best query is the latest corrected query that was checked on
    the engine, else the answer's query if most ratings say correct.
    """
    for rating in reversed(record["feedback"]):
        if rating["corrected_query"] and rating["corrected_query_checked"]:
            source, query = "corrected", rating["corrected_query"]
            break
    else:
        if record["status"] != "correct" or not record["valid"]:
            return None
        source, query = "approved", record["query"]
    return {
        "question": record["question"], "query": query,
        "language": record["language"] or "datalog",
        "engine": record["engine"], "source": source,
    }


def export_records(store: FeedbackStore, engine=None):
    """Yield the answers with their ratings and their training pair."""
    for record in store.answers(engine):
        record["training"] = training_pair(record)
        yield record


def examples_yaml(store: FeedbackStore, engine=None) -> str:
    """
    Return the training pairs as a bank of examples (``examples/*.yaml``),
    one example per question, its latest best query.
    """
    import yaml

    class Dumper(yaml.SafeDumper):
        pass

    Dumper.add_representer(str, lambda dumper, text: dumper.represent_scalar(
        "tag:yaml.org,2002:str", text, style="|" if "\n" in text else None
    ))
    examples = {}
    for record in export_records(store, engine):
        pair = record["training"]
        if pair and pair["question"] not in examples:
            examples[pair["question"]] = {
                "question": pair["question"], pair["language"]: pair["query"],
            }
    header = (
        "# Examples approved or corrected on a neurolang-feedback-server"
        + (f" ({engine})" if engine else "") + ".\n"
    )
    return header + yaml.dump(
        {"examples": list(examples.values())}, Dumper=Dumper,
        sort_keys=False, allow_unicode=True,
    )


class RateLimit:
    """At most *per_minute* requests of each address in a minute."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.times = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, address: str) -> bool:
        now = time.monotonic()
        with self.lock:
            times = self.times[address]
            while times and now - times[0] > 60:
                times.popleft()
            if len(times) >= self.per_minute:
                return False
            times.append(now)
            return True


class Handler(BaseHTTPRequestHandler):
    """The requests of the feedback server (see the module)."""

    server_version = "NeuroLangFeedback/1"

    def _send(self, status: int, body, content_type="application/json"):
        if not isinstance(body, (str, bytes)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Access-Control-Allow-Origin", self.server.cors_origin
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Authorization"
        )
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str):
        self._send(status, {"error": message})

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        url = urlparse(self.path)
        params = {k: v[-1] for k, v in parse_qs(url.query).items()}
        store = self.server.store
        if url.path == "/":
            self._send(200, REVIEW_PAGE, "text/html; charset=utf-8")
        elif url.path == "/api/answers":
            status = params.get("status") or None
            if status and status not in STATUSES:
                return self._error(400, f"status: one of {STATUSES}")
            try:
                limit = min(int(params.get("limit", PAGE_SIZE)), 200)
                offset = max(int(params.get("offset", 0)), 0)
            except ValueError:
                return self._error(400, "limit and offset are integers")
            engine = params.get("engine")
            self._send(200, {
                "answers": store.answers(engine, status, limit, offset),
                "total": store.count(engine, status),
            })
        elif url.path == "/api/export.jsonl":
            token = self.server.admin_token
            given = self.headers.get("Authorization", "")
            if not token:
                return self._error(403, "export is disabled: no admin token")
            if not hmac.compare_digest(given, f"Bearer {token}"):
                return self._error(401, "admin token required")
            self._send(200, "".join(
                json.dumps(record) + "\n"
                for record in export_records(store, params.get("engine"))
            ), "application/x-ndjson")
        else:
            self._error(404, "not found")

    def do_POST(self):
        if not self.server.rate_limit.allow(self.client_address[0]):
            return self._error(429, "too many requests")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._error(411, "Content-Length required")
        if length > self.server.max_bytes:
            self.close_connection = True
            return self._error(413, "request too large")
        try:
            data = json.loads(self.rfile.read(length) or b"null")
            path = urlparse(self.path).path
            match = _FEEDBACK_PATH.match(path)
            if path == "/api/answers":
                self._send(201, {"id": self.server.store.add_answer(data)})
            elif match:
                self._send(201, {"id": self.server.store.add_feedback(
                    match.group(1), data
                )})
            else:
                self._error(404, "not found")
        except ValueError as exc:  # JSONDecodeError included
            self._error(400, f"invalid request: {exc}")
        except KeyError:
            self._error(404, "no such answer")


def make_server(
    db, host="127.0.0.1", port=8765, cors_origin="*", admin_token=None,
    max_bytes=MAX_BYTES, per_minute=POSTS_PER_MINUTE,
) -> ThreadingHTTPServer:
    """Return the feedback server on the SQLite database *db*."""
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.store = FeedbackStore(db)
    server.cors_origin = cors_origin
    server.admin_token = admin_token
    server.max_bytes = max_bytes
    server.rate_limit = RateLimit(per_minute)
    return server


def main(argv=None) -> None:
    """``neurolang-feedback-server [export] --db PATH ...``"""
    argv = sys.argv[1:] if argv is None else argv
    export = bool(argv) and argv[0] == "export"
    parser = argparse.ArgumentParser(
        prog="neurolang-feedback-server" + (" export" if export else ""),
        description=(
            "Export the rated answers of a feedback server" if export else
            "Collect the answers of the NeuroLang query assistant and "
            "their ratings"
        ),
    )
    parser.add_argument("--db", required=True, help="SQLite database")
    if export:
        parser.add_argument(
            "--format", choices=("jsonl", "examples"), default="jsonl",
            help="jsonl: every answer, its steps and ratings; examples: a "
                 "bank of examples of the approved and corrected queries",
        )
        parser.add_argument("--engine", help="only the answers on ENGINE")
        parser.add_argument("-o", "--output", help="file (default: stdout)")
        args = parser.parse_args(argv[1:])
        if not Path(args.db).exists():
            parser.error(f"no database {args.db}")
        store = FeedbackStore(args.db)
        text = examples_yaml(store, args.engine) if (
            args.format == "examples"
        ) else "".join(
            json.dumps(record) + "\n"
            for record in export_records(store, args.engine)
        )
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--cors-origin", default="*",
        help="the origin of the pages allowed to post (default: any)",
    )
    parser.add_argument(
        "--admin-token-env", default=ADMIN_TOKEN_ENV,
        help="environment variable holding the token of the export "
             f"(default: {ADMIN_TOKEN_ENV}); unset, the export is disabled",
    )
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument(
        "--rate", type=int, default=POSTS_PER_MINUTE,
        help="most posts a minute from an address",
    )
    args = parser.parse_args(argv)
    server = make_server(
        args.db, args.host, args.port, args.cors_origin,
        os.environ.get(args.admin_token_env), args.max_bytes, args.rate,
    )
    print(f"Feedback server on http://{args.host}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


REVIEW_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuroLang assistant answers</title>
<style>
body { font: 15px/1.45 system-ui, sans-serif; margin: 0 auto;
       max-width: 60rem; padding: 1rem; color: #1b2733; }
header, .bar { display: flex; gap: .5rem; align-items: center;
               flex-wrap: wrap; }
.answer { border-top: 1px solid #c9d3dd; padding: .8rem 0; }
.meta { color: #5b6b7b; font-size: 85%; }
pre, textarea { font: 13px/1.4 ui-monospace, monospace; background: #f3f6f9;
                padding: .5rem; overflow-x: auto; width: 100%;
                box-sizing: border-box; }
.status-correct { color: #19703a; } .status-incorrect { color: #b3261e; }
button { cursor: pointer; }
</style></head><body>
<header><h1>Assistant answers</h1>
<label>Status <select id="status"><option value="">all</option>
<option>unrated</option><option>correct</option><option>incorrect</option>
<option>unsure</option></select></label>
<label>Engine <input id="engine" size="12"></label>
<label>Your name <input id="user" size="12"></label>
<button id="load">Show</button></header>
<p class="bar"><button id="prev">&larr;</button><span id="page"></span>
<button id="next">&rarr;</button></p>
<div id="answers"></div>
<template id="answer"><div class="answer">
<h3 class="question"></h3><div class="meta"></div>
<pre class="query"></pre><p class="explanation"></p>
<details><summary>Rows</summary><pre class="rows"></pre></details>
<details><summary>Steps</summary><pre class="steps"></pre></details>
<ul class="ratings"></ul>
<details><summary>Rate or correct</summary>
<p>Corrected query (optional)</p><textarea class="corrected" rows="5">
</textarea><p>Comment</p><textarea class="comment" rows="2"></textarea>
<p class="bar"><button data-verdict="correct">Correct</button>
<button data-verdict="incorrect">Incorrect</button>
<button data-verdict="unsure">Unsure</button><span class="sent"></span></p>
</details></div></template>
<script>
const $ = (id) => document.getElementById(id);
let offset = 0;
const size = 20;
$("user").value = localStorage.getItem("neurolang-user") || "";
$("user").onchange = () => localStorage.setItem("neurolang-user",
                                                $("user").value);
function text(node, selector, value) {
  node.querySelector(selector).textContent = value || "";
}
async function rate(record, card, verdict) {
  const body = {verdict, user: $("user").value,
                comment: card.querySelector(".comment").value};
  const corrected = card.querySelector(".corrected").value.trim();
  if (corrected && corrected !== (record.query || "").trim()) {
    body.corrected_query = corrected;
    body.corrected_query_checked = false;
  }
  const response = await fetch(`api/answers/${record.id}/feedback`,
    {method: "POST", headers: {"Content-Type": "application/json"},
     body: JSON.stringify(body)});
  text(card, ".sent", response.ok ? "Thank you." :
       (await response.json()).error);
}
function show(record) {
  const card = $("answer").content.firstElementChild.cloneNode(true);
  text(card, ".question", record.question);
  text(card, ".meta", [record.created, record.engine, record.provider,
    record.model, record.strategy, record.valid ? "valid" : "invalid",
    record.status].filter(Boolean).join(" · "));
  card.querySelector(".meta").classList.add("status-" + record.status);
  text(card, ".query", record.query || record.error);
  text(card, ".explanation", record.explanation);
  text(card, ".rows", record.rows);
  text(card, ".steps", record.steps.map((s) =>
    `${s.thought ? s.thought + "\\n" : ""}-> ${s.action} ` +
    `${JSON.stringify(s.arguments)}\\n${s.observation}`).join("\\n\\n"));
  for (const r of record.feedback) {
    const item = document.createElement("li");
    item.textContent = `${r.verdict} by ${r.user || "anonymous"}` +
      (r.comment ? `: ${r.comment}` : "") +
      (r.corrected_query ? ` — corrected (${r.corrected_query_checked ?
        "checked" : "unchecked"}): ${r.corrected_query}` : "");
    card.querySelector(".ratings").append(item);
  }
  card.querySelector(".corrected").value = record.query || "";
  for (const button of card.querySelectorAll("[data-verdict]")) {
    button.onclick = () => rate(record, card, button.dataset.verdict);
  }
  $("answers").append(card);
}
async function load() {
  const params = new URLSearchParams({limit: size, offset,
    status: $("status").value, engine: $("engine").value});
  const data = await (await fetch(`api/answers?${params}`)).json();
  $("answers").replaceChildren();
  data.answers.forEach(show);
  $("page").textContent = data.total ? `${offset + 1}–${offset +
    data.answers.length} of ${data.total}` : "No answers";
}
$("load").onclick = () => { offset = 0; load(); };
$("prev").onclick = () => { offset = Math.max(0, offset - size); load(); };
$("next").onclick = () => { offset += size; load(); };
load();
</script></body></html>
"""
