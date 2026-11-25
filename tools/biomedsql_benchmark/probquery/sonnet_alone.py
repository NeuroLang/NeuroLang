"""Arm S0: one call per question with the schema description and the
uncertainty paragraph of S1, no database and no tool."""

import time

from arms import (
    ANSWER_SQL, TEMPERATURE, UNCERTAINTY, openrouter_client, parse_answer,
    schema, usage_of,
)

ROLE = (
    "You answer questions about a database whose description follows. You "
    "cannot query it: answer from the description and what you know."
)


def run(question):
    from neurolang_apps.utils.llm.providers import LLMError

    _, client = openrouter_client()
    messages = [
        {"role": "system", "content": "\n\n".join([
            ROLE, "# Database\n" + schema(), "# Uncertainty\n" + UNCERTAINTY,
            ANSWER_SQL,
        ])},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    rec = {"answer": None, "error": "", "steps": 1, "final_query": None,
           "tools": {"calls": 0, "failed": 0, "seconds": 0.0}}
    try:
        reply = client.complete(messages, None, TEMPERATURE)
        rec["answer"] = parse_answer(reply.text)
        rec["log"] = [{"text": reply.text}]
        if rec["answer"] is None:
            rec["error"] = "no 'ANSWER: <number>' in the reply"
    except LLMError as e:
        rec["error"] = f"LLMError: {e}"
    rec["usage"] = usage_of(client)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec
