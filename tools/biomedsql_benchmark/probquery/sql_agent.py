"""Arm S1: Sonnet 5.5 with one tool, run_sql, over DuckDB.

The database is the twelve tables of ``arms.table_files()``, as views over
their parquet files: the five BiomedSQL tables as the NeuroLang engine reads
them (missing text '', booleans the text 'True'/'False'), and the seven tables
of ``data/``. A query is one SELECT (or WITH ... SELECT) statement; at most
200 rows are shown; errors come back as text; a query is interrupted after
``QUERY_TIMEOUT`` seconds. No Python, no calculator.

A step is one reply of the model (as in NeuroLang's ReactAgent); at most 25.
The answer is the number of the last 'ANSWER: <number>' of a reply without a
tool call; none after 25 steps is no answer.

Needs duckdb, which the project does not depend on: run with
``uv run --no-sync --with duckdb``.
"""

import threading
import time

from arms import (
    ANSWER_SQL, MAX_STEPS, QUERY_TIMEOUT, TEMPERATURE, UNCERTAINTY,
    biomedsql_tables, openrouter_client, parse_answer, query_size, schema,
    table_files,
    usage_of,
)

MAX_ROWS = 200
MAX_OBSERVATION = 30000     # characters of a tool result

ROLE = (
    "You answer questions about a database by querying it with SQL (DuckDB "
    "dialect) through the tool run_sql. You have no other tool: no Python and "
    "no calculator; compute with SQL. Explore the tables as you need, then "
    "give your final answer in a reply without a tool call."
)

TOOL = {
    "name": "run_sql",
    "description": "Run one read-only SQL query (DuckDB dialect) on the "
                   f"database and return its result: the column names and "
                   f"at most {MAX_ROWS} rows, or the error.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string",
                                 "description": "A SELECT statement."}},
        "required": ["query"],
    },
}


def system_prompt():
    return "\n\n".join([
        ROLE, "# Database\n" + schema(), "# Uncertainty\n" + UNCERTAINTY,
        ANSWER_SQL,
    ])


class Database:
    """A read-only DuckDB connection over the parquet files of the tables."""

    def __init__(self):
        import duckdb

        self.duckdb = duckdb
        self.con = duckdb.connect(":memory:", config={"threads": 4})
        big = biomedsql_tables()
        for name, path in table_files().items():
            source = f"read_parquet('{path}')"
            if name in big:
                cols = self.con.execute(
                    f"DESCRIBE SELECT * FROM {source}"
                ).fetchall()
                exprs = []
                for col, kind, *_ in cols:
                    q = f'"{col}"'
                    if kind == "VARCHAR":
                        exprs.append(f"coalesce({q}, '') AS {q}")
                    elif kind == "BOOLEAN":
                        exprs.append(
                            f"CASE WHEN {q} THEN 'True' WHEN NOT {q} THEN "
                            f"'False' ELSE '' END AS {q}"
                        )
                    else:
                        exprs.append(q)
                select = f"SELECT {', '.join(exprs)} FROM {source}"
            else:
                select = f"SELECT * FROM {source}"
            self.con.execute(f"CREATE VIEW {name} AS {select}")
        # No file but the tables' can be read, and the setting is locked.
        dirs = sorted({str(p.parent) + "/" for p in table_files().values()})
        self.con.execute(f"SET allowed_directories = {dirs!r}")
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    def run(self, query):
        """The result of *query* as text, or its error."""
        try:
            statements = self.con.extract_statements(query)
        except self.duckdb.Error as e:
            return f"Error: {e}"
        if len(statements) != 1:
            return "Error: give exactly one SQL statement."
        if statements[0].type != self.duckdb.StatementType.SELECT:
            return "Error: the database is read-only; only SELECT queries run."
        timer = threading.Timer(QUERY_TIMEOUT, self.con.interrupt)
        timer.start()
        try:
            cursor = self.con.execute(query)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchmany(MAX_ROWS + 1)
        except self.duckdb.InterruptException:
            return f"Error: the query took more than {QUERY_TIMEOUT} s."
        except self.duckdb.Error as e:
            return f"Error: {e}"
        finally:
            timer.cancel()
        more = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]
        lines = ["\t".join(columns)]
        lines += ["\t".join(repr(v) if isinstance(v, float) else str(v)
                            for v in r) for r in rows]
        head = (f"More than {MAX_ROWS} rows; the first {MAX_ROWS}:"
                if more else f"{len(rows)} rows:")
        text = head + "\n" + "\n".join(lines)
        if len(text) > MAX_OBSERVATION:
            text = text[:MAX_OBSERVATION] + "\n[cut]"
        return text


def run(question, db):
    """Answer *question*: a record of the answer, steps, usage and log."""
    from neurolang_apps.utils.llm.providers import LLMError

    _, client = openrouter_client()
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": question},
    ]
    log, answer, error = [], None, ""
    tools = {"calls": 0, "failed": 0, "seconds": 0.0}
    final_query = None      # the last query that ran without error
    t0 = time.time()
    steps = 0
    while steps < MAX_STEPS:
        steps += 1
        try:
            reply = client.complete(messages, [TOOL], TEMPERATURE)
        except LLMError as e:
            error = f"LLMError: {e}"
            break
        if not reply.tool_calls:
            messages.append({"role": "assistant", "content": reply.text})
            log.append({"step": steps, "text": reply.text})
            answer = parse_answer(reply.text)
            if answer is not None:
                break
            messages.append({
                "role": "user",
                "content": "Call run_sql, or give your final answer as "
                           "'ANSWER: <number>'.",
            })
            continue
        messages.append({"role": "assistant", "content": reply.text,
                         "tool_calls": reply.tool_calls})
        for call in reply.tool_calls:
            tools["calls"] += 1
            t1 = time.perf_counter()
            if call.name != "run_sql" or "query" not in call.arguments:
                result = "Error: the only tool is run_sql(query)."
            else:
                result = db.run(call.arguments["query"])
            tools["seconds"] += time.perf_counter() - t1
            if result.startswith("Error:"):
                tools["failed"] += 1
            else:
                final_query = call.arguments["query"]
            log.append({"step": steps, "text": reply.text,
                        "tool": call.name, "arguments": call.arguments,
                        "result": result[:2000]})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "name": call.name, "content": result})
    if answer is None and not error:
        error = f"no answer after {MAX_STEPS} steps"
    tools["seconds"] = round(tools["seconds"], 2)
    return {
        "answer": answer, "steps": len(client.calls), "error": error,
        "tools": tools, "final_query": final_query,
        "query_size": query_size(final_query, sql=True),
        "usage": usage_of(client), "seconds": round(time.time() - t0, 1),
        "log": log,
    }
