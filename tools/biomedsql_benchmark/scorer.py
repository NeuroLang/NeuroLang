"""Scoring of an answer to a BiomedSQL item: a strict mechanical match and a judge.

An answer is a list of relations, each ``(columns, rows)`` with canonical string
values (``common.canon``); a program with several named answers gives several
relations, a boolean answer none.

* ``strict``: the answer reproduces BiomedSQL's stored result exactly, as a
  set of rows over all its columns but UUID, under some assignment of the
  answer's columns to them (names and order do not matter, extra columns do not
  pass). Empty result: correct iff the answer is empty. Truncated result (as
  many rows as the LIMIT): correct iff all its rows are in the answer.
* ``judge``: a language model, blind to the query and to the system, compares
  the answer table with the stored result under the rubric below, and may
  accept a missing column that the question does not need, and different but
  equivalent shapes. The rubric was validated on controls of the development
  items (``validate_judge.py``) and frozen before any test run.
"""

import itertools
import json
import os
import threading
import urllib.request

from common import HERE, table_text

MODEL = "claude-sonnet-5-5"

JUDGE_PROMPT = """You grade the answer of a system to a question about a biomedical database.
You are given the question, a reference answer in prose, the reference result table (the rows returned by the correct SQL query, an internal row id removed) and the result the system returned.

Decide whether the system's result correctly answers the question, that is whether it gives the information of the reference table, with the same rows:
- Same entities and values; numbers agree to 3 significant digits.
- Extra columns, other column names or a different column order are fine. A column of the reference table that the system lacks is fine only if the question does not need it (an identifier or a context column).
- A row of the reference table that the system lacks, or a row the system has that the reference table does not have, makes it incorrect. Repeated rows do not matter.
- If the reference table is empty, the system is correct only if its result is empty, or says there is none.
- If the reference table is marked as truncated, it is the first rows of a longer ordered list: the system is correct if it contains all of them, and may contain more.
- A bare yes/no (true or false) is not an answer: the system must give the rows or values of the reference table, even for a yes/no question. The one exception is an empty reference table, where \"false\" is correct.

First compare the two tables row by row and value by value, and list every difference you find (a missing row, an extra row, a different number or text). Then decide.
Reply with JSON only, as {{"differences": "what differs, or none", "verdict": "correct" or "incorrect"}}.

# Question
{question}

# Reference answer in prose
{answer}

# Reference result table{truncated}
{reference}

# System's result
{result}
"""


def strict(columns, rows, truncated, answers):
    """Mechanical match of the answer relations with the gold table."""
    gold = set(rows)
    if not gold:
        return all(not rel_rows for _, rel_rows in answers)
    k = len(columns)
    gold_cols = [{r[i] for r in gold} for i in range(k)]
    for _, rel_rows in answers:
        if not rel_rows or len(rel_rows[0]) < k:
            continue
        width = len(rel_rows[0])
        cols = [{r[j] for r in rel_rows} for j in range(width)]
        candidates = []
        for gc in gold_cols:
            ok = [
                j
                for j in range(width)
                if (gc <= cols[j] if truncated else gc == cols[j])
            ]
            if not ok:
                break
            candidates.append(ok)
        else:
            for assign in itertools.product(*candidates):
                if len(set(assign)) < k:
                    continue
                projected = {tuple(r[j] for j in assign) for r in rel_rows}
                if gold <= projected if truncated else projected == gold:
                    return True
    return False


def result_text(answers, limit=25):
    if not answers:
        return "(no result)"
    return "\n\n".join(table_text(cols, rows, limit) for cols, rows in answers)


def judge_prompt(item, columns, rows, truncated, result):
    return JUDGE_PROMPT.format(
        question=item["question"],
        answer=item["answer"],
        truncated=" (TRUNCATED)" if truncated else "",
        reference=table_text(columns, rows),
        result=result,
    )


def call_judge(prompt, cache=None):
    """The judge's verdict ('correct'/'incorrect') and reason; cached by prompt."""
    import hashlib

    key = hashlib.sha256(prompt.encode()).hexdigest()
    cache_path = (
        (HERE / "judge_cache" / f"{key}.json")
        if cache is None
        else cache / f"{key}.json"
    )
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            pass
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(
            {
                "model": MODEL,
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode(),
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_INRIA_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    for attempt in range(4):
        try:
            reply = json.load(urllib.request.urlopen(request, timeout=120))
            break
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
    text = "".join(
        b.get("text", "") for b in reply["content"] if b["type"] == "text"
    )
    out = {"verdict": "unparsed", "reason": text[-300:]}
    start = text.rfind('{"differences"')
    if start < 0:
        start = text.rfind("{")
    try:
        parsed = json.loads(text[start : text.rindex("}") + 1])
        if parsed["verdict"] in ("correct", "incorrect"):
            out = {
                "verdict": parsed["verdict"],
                "reason": parsed.get("differences", ""),
            }
    except Exception:  # noqa: BLE001
        pass
    out["usage"] = reply["usage"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(
        f".{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(json.dumps(out))
    os.replace(temporary, cache_path)
    return out
