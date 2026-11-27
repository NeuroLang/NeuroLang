"""Arm S2: NeuroLang's ReAct agent (``ReactAgent``, Datalog) on the engine of
``engine.py`` (the same twelve tables, the uncertain ones as probabilistic
facts, lifted solvers only).

The agent is the repository's, unchanged: its tools (list_predicates,
describe_predicate, search_values, check_query, run_query, final_answer), its
system prompt and Datalog reference, with no worked examples, no verification
round and no rule comments; at most 25 steps (model replies); the model and
route of the other arms (``arms.openrouter_client``).

Its prompt is given the schema text of the other arms (the uncertain tables
without their column prob) as the engine's description, and as the guidance:
the Datalog language notes of the ``biomedsql`` engine's guidance, the
uncertainty paragraph (the uncertain tables as probabilistic relations) and
the form of the answer. The rest of the ``biomedsql`` guidance is left out:
it sets BiomedSQL's answer conventions (rows of evidence, never a bare
value) and the use of the qtl_* / selected_qtl / support_prior relations,
which this engine does not have.

The answer is the final query, run again: one row, its last numeric value. A
query refused or failing, several rows, no final answer within 25 steps
(``out_of_steps``: the agent's fallback, its last query giving rows, is
recorded but not scored) is no answer.
"""

import os
import signal
import time

import yaml

from arms import (
    ANSWER_DATALOG, MAX_STEPS, QUERY_TIMEOUT, REQUEST_TIMEOUT, TEMPERATURE,
    UNCERTAINTY_DATALOG, openrouter_client, query_size, schema, usage_of,
)
from build_tables import ENGINES


def language_notes():
    """The sentences of the biomedsql guidance about the language."""
    text = yaml.safe_load(open(ENGINES / "biomedsql/guidance.yaml"))[
        "assistant_guidance"
    ]
    start = text.index("In the language:")
    end = text.index("A question asking how consistently")
    return " ".join(text[start:end].split())


# Run B of PROTOCOL.md amendment 5: three rules, with PQ_GUIDANCE=v2.
RULES_V2 = (
    "Write the whole question as one PROB[...] query: put every condition, "
    "join and quantifier of the question in the rules under it, and let the "
    "engine combine the probabilities. For 'given' or 'conditional on', use "
    "the conditional form PROB[event // condition]. Never combine "
    "probabilities by arithmetic or aggregation, and never write a "
    "probability, or an entity you found by looking at the data, as a "
    "constant: if a query is refused, rewrite the rules, not the arithmetic."
    " For an expected value, write EXPECT[f | event] = e, the expected value "
    "of f summed over the answers of the event; EXPECT[1 | event] = n is the "
    "expected number of answers."
)
GUIDANCE = "\n".join(
    [language_notes(), UNCERTAINTY_DATALOG, ANSWER_DATALOG]
    + ([RULES_V2] if os.environ.get("PQ_GUIDANCE") == "v2" else [])
)


# A tool call failed when a check or a run of its query failed (a parse,
# solver or execution error, a timeout), or when the tool itself refused its
# call (unknown tool, wrong arguments). A final_answer asking to confirm an
# empty answer is not a failure.
_REFUSED = ("Error:", "Wrong arguments", "Unknown tool",
            "The arguments are not")


def _instrumented_agent():
    """ReactAgent, its tool calls timed and their failures counted (the
    agent module's check_query and run_preview are wrapped in this process
    to see whether a call's query failed)."""
    from neurolang_apps.utils.llm import agent as agent_module

    if not hasattr(agent_module, "_probquery_failed"):
        agent_module._probquery_failed = [False]
        for name in ("check_query", "run_preview"):
            original = getattr(agent_module, name)

            def wrapped(*args, _original=original, **kwargs):
                out = _original(*args, **kwargs)
                if not out.ok:
                    agent_module._probquery_failed[0] = True
                return out

            setattr(agent_module, name, wrapped)
    flag = agent_module._probquery_failed

    class Agent(agent_module.ReactAgent):
        tools = None

        def _call(self, result, toolbox, thought, name, arguments):
            flag[0] = False
            t0 = time.perf_counter()
            try:
                return super()._call(result, toolbox, thought, name,
                                     arguments)
            finally:
                self.tools["seconds"] += time.perf_counter() - t0
                self.tools["calls"] += 1
                observation = result.steps[-1].observation if (
                    result.steps and result.steps[-1].action == name
                ) else ""
                if flag[0] or observation.startswith(_REFUSED):
                    self.tools["failed"] += 1

    return Agent


class Alarm(Exception):
    pass


def _alarm(*_):
    raise Alarm()


class DatalogArm:
    def __init__(self):
        from neurolang_apps.utils.llm.config import AssistantConfig

        from engine import build, columns

        self.nl = build()
        self.columns = columns()
        self.config = AssistantConfig(
            strategy="react", language="datalog", max_steps=MAX_STEPS,
            verify_rounds=0, comment_rules=False, temperature=TEMPERATURE,
            samples=1, timeout=REQUEST_TIMEOUT, query_timeout=QUERY_TIMEOUT,
            query_memory_mb=30000.0,
        )

    def run(self, question):
        provider, client = openrouter_client()
        agent = _instrumented_agent()(
            self.nl, provider, self.config, "datalog", engine_name="",
            client=client, loaded=self.columns, examples=[],
            description=schema(datalog=True), guidance=GUIDANCE,
        )
        agent.tools = {"calls": 0, "failed": 0, "seconds": 0.0}
        t0 = time.time()
        rec = {"answer": None, "error": "", "final_query": None}
        try:
            res = agent.run(question)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            res = None
        if res is not None:
            rec.update(
                query=res.query, valid=res.valid,
                out_of_steps=res.out_of_steps, error=res.error,
                log=[{"thought": s.thought[:2000], "tool": s.action,
                      "arguments": s.arguments,
                      "result": s.observation[:2000]} for s in res.steps],
            )
            rec["final_query"] = res.query or None
            rec["query_size"] = query_size(res.query)
            if res.valid and not res.out_of_steps:
                t1 = time.perf_counter()
                rec.update(self.evaluate(res.query))
                # Running the final query again, outside the agent's time.
                rec["evaluate_seconds"] = round(time.perf_counter() - t1, 2)
            elif res.out_of_steps:
                rec["error"] = f"no final answer after {MAX_STEPS} steps"
        agent.tools["seconds"] = round(agent.tools["seconds"], 2)
        rec["tools"] = agent.tools
        rec["steps"] = len(client.calls)
        rec["usage"] = usage_of(client)
        rec["seconds"] = round(time.time() - t0, 1)
        return rec

    def evaluate(self, query):
        """The answer of the final query: its one row's last number."""
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(QUERY_TIMEOUT)
        try:
            with self.nl.scope:
                result = self.nl.execute_datalog_program(query)
            if isinstance(result, dict):
                return {"error": "several named answers"}
            if isinstance(result, bool):
                return {"error": "a yes/no answer, not a number"}
            rows = [tuple(r) for r in result]
        except Alarm:
            return {"error": "the final query timed out"}
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        finally:
            signal.alarm(0)
        out = {"rows": [list(map(str, r)) for r in rows[:5]],
               "n_rows": len(rows)}
        if len(rows) != 1:
            out["error"] = f"{len(rows)} rows, not one"
            return out
        numbers = [v for v in rows[0] if _is_number(v)]
        if not numbers:
            out["error"] = "no number in the answer row"
            return out
        out["answer"] = float(numbers[-1])
        return out


def _is_number(v):
    if isinstance(v, bool):
        return False
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return not isinstance(v, str)
