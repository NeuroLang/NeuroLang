import pytest

from ..agent import (
    IclAgent,
    ReactAgent,
    extract_query,
    make_agent,
    parse_text_action,
    parse_verdict,
)
from ..providers import LLMError, Reply
from .fakes import ScriptedClient, assistant_config, provider

PAIN_SQUALL = "obtain every Study that mentions 'pain'."
PAIN_DATALOG = "ans(s) :- mentions(s, 'pain')"


def agent(nl, cls, replies, language="squall", tool_calling="text",
          **options):
    """Return an agent of *cls* answered by the scripted *replies*."""
    steps = []
    instance = cls(
        nl, provider(tool_calling), assistant_config(**options), language,
        client=ScriptedClient(replies), on_step=steps.append,
    )
    return instance, steps


def fenced(query, language="squall"):
    return f"```{language}\n{query}\n```\nStudies about pain."


def test_extract_query():
    assert extract_query(fenced(PAIN_SQUALL), "squall") == PAIN_SQUALL
    text = f"```\n{PAIN_DATALOG}\n```"
    assert extract_query(text, "datalog") == PAIN_DATALOG
    both = f"```python\nx = 1\n```\n```datalog\n{PAIN_DATALOG}\n```"
    assert extract_query(both, "datalog") == PAIN_DATALOG
    assert extract_query("I cannot answer that.", "squall") is None


@pytest.mark.parametrize("text, name, arguments", [
    (
        'Thought: look\nAction: describe_predicate\n'
        'Action Input: {"name": "study"}',
        "describe_predicate", {"name": "study"},
    ),
    (
        "Thought: list\nAction: `list_predicates`\nAction Input: {}\n"
        "Observation: made up",
        "list_predicates", {},
    ),
    (
        'Thought: done\nFinal Answer: {"query": "obtain every Study.", '
        '"explanation": "all"}',
        "final_answer", {"query": "obtain every Study.", "explanation": "all"},
    ),
    (
        fenced(PAIN_SQUALL), "final_answer",
        {"query": PAIN_SQUALL, "explanation": ""},
    ),
    (
        'Action: check_query\nAction Input: {"query": "a\\nb"}',
        "check_query", {"query": "a\nb"},
    ),
    (
        'Action: check_query\nAction Input: {\n  "query": "a\nb"\n}',
        "check_query", {"query": "a\nb"},
    ),
])
def test_parse_text_action(text, name, arguments):
    action = parse_text_action(text, "squall")
    assert not action.error
    assert (action.name, action.arguments) == (name, arguments)


def test_parse_text_action_errors():
    assert "no action" in parse_text_action("Hmm.", "squall").error
    action = parse_text_action(
        "Action: check_query\nAction Input: query=x", "squall"
    )
    assert "not a JSON object" in action.error
    action = parse_text_action(
        'Action: check_query\nAction Input: {"query": "ans(s', "datalog"
    )
    assert 'Action Input: {"query": "ans(s) :- study(s)"}' in action.error


def test_react_reply_cut_at_token_limit(nl):
    loop = "Thought: write it\nAction: final_answer\nAction Input: " + (
        '{"query": "ans(s) :- mentions(s, \'pain\')' + ", (s == s)" * 50
    )
    replies = [
        Reply(loop, usage={"input": 10, "output": 2048}),
        f'Action: final_answer\nAction Input: {{"query": "{PAIN_DATALOG}"}}',
    ]
    react, steps = agent(nl, ReactAgent, replies, language="datalog")
    result = react.run("Which studies mention pain?")
    assert result.valid and result.query == PAIN_DATALOG
    assert "cut at 2048 tokens" in steps[0].observation
    messages = react.client.requests[1]["messages"]
    assert messages[-2] == {"role": "assistant", "content": "write it"}


def test_icl_first_answer(nl):
    icl, steps = agent(nl, IclAgent, [fenced(PAIN_SQUALL)])
    result = icl.run("Which studies mention pain?")
    assert result.valid and result.query == PAIN_SQUALL
    assert result.explanation == "Studies about pain."
    assert result.calls == 1 and result.usage == {"input": 10, "output": 5}
    assert [s.observation for s in steps] == ["valid"]
    system = icl.client.requests[0]["messages"][0]["content"]
    assert "SQUALL syntax" in system and "mentions(c0, c1)" in system
    assert "fenced code block" in system


def test_icl_repairs_the_query(nl):
    wrong = "obtain every Study that mention 'pain'."
    icl, steps = agent(nl, IclAgent, [fenced(wrong), fenced(PAIN_SQUALL)])
    result = icl.run("Which studies mention pain?")
    assert result.valid and result.query == PAIN_SQUALL
    assert "did you mean mentions" in steps[0].observation
    repair = icl.client.requests[1]["messages"][-1]["content"]
    assert wrong in repair and "did you mean mentions" in repair


def test_icl_gives_up_after_the_repairs(nl):
    wrong = fenced("ans(s) :- mention(s, 'pain')", "datalog")
    icl, _ = agent(nl, IclAgent, [wrong] * 3, language="datalog")
    result = icl.run("Which studies mention pain?")
    assert not result.valid and result.calls == 3
    assert "did you mean mentions" in result.error
    assert result.query == "ans(s) :- mention(s, 'pain')"


def test_icl_without_query(nl):
    icl, _ = agent(nl, IclAgent, ["These predicates cannot answer it."])
    result = icl.run("What is the weather?")
    assert not result.valid and "cannot answer" in result.error


def test_icl_self_consistency_votes_on_the_results(nl):
    replies = [
        fenced("ans(s) :- mentions(s, 'pain')", "datalog"),
        fenced("ans(s) :- mentions(s, 'chronic pain')", "datalog"),
        fenced("ans(x) :- study(x), mentions(x, 'pain')", "datalog"),
    ]
    icl, _ = agent(nl, IclAgent, replies, language="datalog", samples=3)
    result = icl.run("Which studies mention pain?")
    assert result.valid and result.query == "ans(s) :- mentions(s, 'pain')"
    assert result.calls == 3
    assert all(r["temperature"] >= 0.7 for r in icl.client.requests)


def test_react_text_protocol(nl):
    replies = [
        'Thought: find the term\nAction: search_values\nAction Input: '
        '{"predicate": "mentions", "column": "c1", "pattern": "pain"}',
        "Thought: try\nAction: final_answer\nAction Input: "
        '{"query": "obtain every Study that mention \'pain\'.", '
        '"explanation": "x"}',
        "I should give an action.",
        "Thought: run it\nAction: run_query\nAction Input: "
        '{"query": "obtain every Study that mentions \'pain\'."}',
        'Action: final_answer\nAction Input: {"query": '
        '"obtain every Study that mentions \'pain\'.", '
        '"explanation": "Studies mentioning pain."}',
    ]
    react, steps = agent(nl, ReactAgent, replies, max_steps=6)
    result = react.run("Which studies mention pain?")
    assert result.valid and result.query == PAIN_SQUALL
    assert result.explanation == "Studies mentioning pain."
    assert [s.action for s in steps] == [
        "search_values", "final_answer", "", "run_query", "final_answer",
    ]
    assert "mentions(2, 'chronic pain')" in steps[0].observation
    assert "rejected" in steps[1].observation
    assert "no action" in steps[2].observation
    assert "1 rows" in steps[3].observation
    messages = react.client.requests[-1]["messages"]
    assert "Action Input:" in messages[0]["content"]
    assert messages[2]["role"] == "assistant"
    assert messages[3]["content"].startswith("Observation: Facts")
    assert react.client.requests[0]["tools"] is None


def test_react_native_tools(nl):
    replies = [
        [("describe_predicate", {"name": "mentions"}),
         ("search_values", {"predicate": "mentions", "column": "c1",
                            "pattern": "fear"})],
        [("final_answer", {"query": PAIN_DATALOG, "explanation": "pain"})],
    ]
    react, steps = agent(
        nl, ReactAgent, replies, language="datalog", tool_calling="native"
    )
    result = react.run("Which studies mention pain?")
    assert result.valid and result.query == PAIN_DATALOG
    assert [s.action for s in steps] == [
        "describe_predicate", "search_values", "final_answer"
    ]
    request = react.client.requests[1]
    assert request["tools"][0]["name"] == "list_predicates"
    assert "Action Input" not in request["messages"][0]["content"]
    assert [m["role"] for m in request["messages"]] == [
        "system", "user", "assistant", "tool", "tool"
    ]
    assert request["messages"][4]["content"].startswith("Facts with")


def test_react_native_text_answer(nl):
    replies = ["Let me think.", fenced(PAIN_DATALOG, "datalog")]
    react, steps = agent(
        nl, ReactAgent, replies, language="datalog", tool_calling="native"
    )
    result = react.run("Which studies mention pain?")
    assert result.valid and result.query == PAIN_DATALOG
    assert [s.action for s in steps] == ["answer", "final_answer"]


def test_react_tool_errors(nl):
    replies = [
        [("teleport", {})],
        [("describe_predicate", {"nme": "study"})],
        [("search_values", {"predicate": "nothing", "column": "c0",
                            "pattern": "x"})],
        [("check_query", {"query": "ans(s) :- mention(s, 'pain')"})],
    ]
    react, steps = agent(
        nl, ReactAgent, replies, language="datalog", tool_calling="native"
    )
    result = react.run("Which studies mention pain?")
    assert not result.valid and "No valid query after 4 steps" in result.error
    assert result.query == "ans(s) :- mention(s, 'pain')"
    observations = [s.observation for s in steps]
    assert observations[0].startswith("Unknown tool 'teleport'")
    assert observations[1].startswith("Wrong arguments for describe_predicate")
    assert "Unknown predicate 'nothing'" in observations[2]
    assert "did you mean mentions" in observations[3]


def test_react_truncates_observations(nl):
    nl.add_tuple_set([(i, "x" * 50) for i in range(500)], name="big")
    replies = [
        [("run_query", {"query": "ans(a, b) :- big(a, b)"})],
        [("final_answer", {"query": "ans(a, b) :- big(a, b)"})],
    ]
    react, steps = agent(
        nl, ReactAgent, replies, language="datalog", tool_calling="native",
        preview_rows=100,
    )
    assert react.run("Show big").valid
    assert "more characters)" in steps[0].observation


def test_make_agent(nl):
    config = assistant_config(strategy="icl")
    client = ScriptedClient([])
    assert isinstance(
        make_agent(nl, provider(), config, "squall", client=client), IclAgent
    )
    assert isinstance(
        make_agent(nl, provider(), config, "squall", "react", client=client),
        ReactAgent,
    )


def test_examples_and_description_can_be_given(nl):
    from ..examples import Example

    examples = [Example("Which studies?", {"datalog": "ans(s) :- study(s)"})]
    icl = IclAgent(
        nl, provider(), assistant_config(), "datalog",
        engine_name="not-registered", client=ScriptedClient([]),
        examples=examples, description="A tiny engine.",
    )
    system = icl.system_prompt("Which studies?")
    assert "Question: Which studies?\nans(s) :- study(s)" in system
    assert "The engine: A tiny engine." in system


@pytest.mark.parametrize("text, name, arguments", [
    ('**Thought:** look\n**Action:** search_values\n**Action Input:** '
     '{"pattern": "pain"}', "search_values", {"pattern": "pain"}),
    ('Action: `search_values`\naction input: ```json\n'
     '{"pattern": "pain"}\n```',
     "search_values", {"pattern": "pain"}),
    ('Thought: x\nAction: search_values({"pattern": "pain"})',
     "search_values", {"pattern": "pain"}),
    ('### Action: run_query\n- Action_Input: {"query": "ans(x) :- p(x)"}',
     "run_query", {"query": "ans(x) :- p(x)"}),
])
def test_parse_text_action_formatting_slips(text, name, arguments):
    action = parse_text_action(text, "datalog")
    assert not action.error
    assert (action.name, action.arguments) == (name, arguments)


def test_react_answers_the_last_query_with_rows_when_out_of_steps(nl):
    replies = [
        'Action: run_query\nAction Input: {"query": "ans(s) :- study(s)"}',
        'Action: run_query\nAction Input: {"query": '
        '"ans(s) :- mentions(s, \'zzz\')"}',
        "Action: list_predicates\nAction Input: {}",
    ]
    react, _ = agent(nl, ReactAgent, replies, language="datalog", max_steps=3)
    result = react.run("Which studies?")
    assert result.valid and result.query == "ans(s) :- study(s)"
    assert "after 3 steps" in result.explanation


def test_final_answer_runs_the_query(nl):
    replies = [
        'Action: final_answer\nAction Input: {"query": '
        '"ans(s) :- mentions(s, \'zzz\')"}',
        'Action: final_answer\nAction Input: {"query": '
        '"ans(s) :- mentions(s, \'zzz\')"}',
    ]
    react, steps = agent(nl, ReactAgent, replies, language="datalog")
    result = react.run("Which studies mention zzz?")
    assert "gives no rows" in steps[0].observation
    assert result.valid and steps[1].observation == "Accepted."


def test_fallback_prefers_the_most_detailed_query(nl):
    replies = [
        'Action: run_query\nAction Input: {"query": '
        '"ans(s) :- study(s), mentions(s, \'pain\')"}',
        'Action: run_query\nAction Input: {"query": '
        '"ans(s, t) :- mentions(s, t)"}',
    ]
    react, _ = agent(nl, ReactAgent, replies, language="datalog", max_steps=2)
    result = react.run("Which studies mention pain?")
    assert result.query == "ans(s) :- study(s), mentions(s, 'pain')"


def test_empty_final_answer_recalls_the_query_with_rows(nl):
    replies = [
        'Action: run_query\nAction Input: {"query": '
        '"ans(s) :- mentions(s, \'pain\')"}',
        'Action: final_answer\nAction Input: {"query": '
        '"ans(s) :- mentions(s, \'zzz\')"}',
        'Action: final_answer\nAction Input: {"query": '
        '"ans(s) :- mentions(s, \'pain\')"}',
    ]
    react, steps = agent(nl, ReactAgent, replies, language="datalog")
    result = react.run("Which studies mention pain?")
    assert "Your query that gave rows was:\nans(s) :- mentions(s, 'pain')" \
        in steps[1].observation
    assert result.query == "ans(s) :- mentions(s, 'pain')"


def test_engine_guidance_is_in_the_prompt(nl, monkeypatch):
    from neurolang_apps.utils import engine_registry

    monkeypatch.setattr(
        engine_registry, "list_engine_names", lambda: ["guided"]
    )
    monkeypatch.setattr(engine_registry, "get_engine_config", lambda *a: {
        "description": "Studies and terms.",
        "assistant_guidance": "Answer with   probabilities.",
    })
    react = ReactAgent(
        nl, provider(), assistant_config(), "datalog", engine_name="guided",
        client=ScriptedClient([]),
    )
    prompt = react.system_prompt("Which studies mention pain?")
    assert "The engine: Studies and terms." in prompt
    assert "# How to answer on this engine\nAnswer with probabilities." \
        in prompt
    given = ReactAgent(
        nl, provider(), assistant_config(), "datalog", engine_name="guided",
        client=ScriptedClient([]), guidance="",
    )
    assert "How to answer" not in given.system_prompt("Which studies?")


def answer(query, explanation="x"):
    """Return a text protocol reply giving *query* as the final answer."""
    return (
        'Action: final_answer\nAction Input: {"query": "'
        + query.replace("'", "\\'").replace('"', '\\"')
        + f'", "explanation": "{explanation}"}}'
    ).replace("\\'", "'")


ALL_STUDIES = "ans(s) :- study(s)"


@pytest.mark.parametrize("text, verdict, reason", [
    ("VERDICT: yes\nIt lists the studies.", True, "It lists the studies."),
    ("**VERDICT: No** — it lists every study.", False,
     "it lists every study."),
    ("No. The query ignores pain.", False, "The query ignores pain."),
    ("I am not sure.", None, "I am not sure."),
    ("The studies are those mentioning pain. VERDICT: no? No, the region "
     "is right.\nVERDICT: yes", True,
     "The studies are those mentioning pain. VERDICT: no? No, the region "
     "is right."),
])
def test_parse_verdict(text, verdict, reason):
    assert parse_verdict(text) == (verdict, reason)


def test_react_verifies_the_final_answer(nl):
    react, steps = agent(nl, ReactAgent, [
        answer(PAIN_DATALOG),
        "The studies that mention the term pain.",
        "VERDICT: yes\nThey are the studies about pain.",
    ], language="datalog", verify_rounds=1)
    result = react.run("Which studies mention pain?")
    assert result.valid and result.query == PAIN_DATALOG
    assert result.verified is True
    assert result.back_translation == "The studies that mention the term pain."
    assert result.calls == 3
    assert steps[-1].action == "verify"
    assert "Reads back as: The studies" in steps[-1].observation
    assert "Answers the question: yes" in steps[-1].observation
    translation, judgement = react.client.requests[1:]
    # The query is translated back without the question.
    assert "Which studies mention pain?" not in str(translation["messages"])
    assert PAIN_DATALOG in translation["messages"][1]["content"]
    assert "1 rows" in translation["messages"][1]["content"]
    judged = judgement["messages"][1]["content"]
    assert "Which studies mention pain?" in judged
    assert "The engine's definitions are the" in (
        judgement["messages"][0]["content"]
    )
    assert "# Predicates\n- mentions(" in judged
    assert "The studies that mention the term pain." in judged


def test_react_corrects_a_rejected_answer(nl):
    react, steps = agent(nl, ReactAgent, [
        answer(ALL_STUDIES),
        "Every study of the database.",
        "VERDICT: no\nIt does not restrict the studies to pain.",
        answer(PAIN_DATALOG),
        "The studies that mention pain.",
        "VERDICT: yes",
    ], language="datalog", verify_rounds=1)
    result = react.run("Which studies mention pain?")
    assert result.valid and result.query == PAIN_DATALOG
    assert result.verified is True
    rejection = react.client.requests[3]["messages"][-1]["content"]
    assert rejection.startswith("Observation: The answer is rejected")
    assert "Every study of the database." in rejection
    assert "does not restrict the studies to pain" in rejection
    assert [s.action for s in steps] == [
        "final_answer", "verify", "final_answer", "verify"
    ]


def test_react_accepts_once_the_rejections_are_spent(nl):
    react, _ = agent(nl, ReactAgent, [
        answer(ALL_STUDIES),
        "Every study.", "VERDICT: no\nNot about pain.",
        answer(ALL_STUDIES),
        "Every study.", "VERDICT: no\nStill not about pain.",
    ], language="datalog", verify_rounds=1)
    result = react.run("Which studies mention pain?")
    assert result.valid and result.query == ALL_STUDIES
    assert result.verified is False
    assert result.back_translation == "Every study."


def test_react_verification_off_or_failing(nl):
    react, steps = agent(nl, ReactAgent, [answer(PAIN_DATALOG)],
                         language="datalog", verify_rounds=0)
    result = react.run("Which studies mention pain?")
    assert result.valid and result.verified is None and result.calls == 1

    class Unreachable(ScriptedClient):
        """Gives the final answer, then fails as a provider does."""

        def complete(self, messages, tools=None, temperature=0.0):
            if self.replies:
                return super().complete(messages, tools, temperature)
            raise LLMError("fake: HTTP 503")

    steps = []
    react = ReactAgent(
        nl, provider(), assistant_config(verify_rounds=1), "datalog",
        client=Unreachable([answer(PAIN_DATALOG)]), on_step=steps.append,
    )
    result = react.run("Which studies mention pain?")
    assert result.valid and result.verified is None
    assert "verification failed: fake: HTTP 503" in steps[-1].observation


def test_react_unclear_verdict_accepts(nl):
    react, _ = agent(nl, ReactAgent, [
        answer(PAIN_DATALOG), "Studies about pain.", "Hard to say.",
    ], language="datalog", verify_rounds=1)
    result = react.run("Which studies mention pain?")
    assert result.valid and result.verified is None
    assert result.back_translation == "Studies about pain."


def test_verification_reads_the_engine_description_and_guidance(nl):
    steps = []
    react = ReactAgent(
        nl, provider(), assistant_config(verify_rounds=1), "datalog",
        client=ScriptedClient([
            answer(PAIN_DATALOG), "Studies about pain.", "VERDICT: yes",
        ]),
        on_step=steps.append, description="Terms are lower case.",
        guidance="Answer with probabilities.",
    )
    react.run("Which studies mention Pain?")
    translation, judgement = react.client.requests[1:]
    for request in (translation, judgement):
        assert "# The engine\nTerms are lower case." in (
            request["messages"][1]["content"]
        )
    guidance = "# How to answer on this engine\nAnswer with probabilities."
    assert guidance in judgement["messages"][1]["content"]
    assert guidance not in translation["messages"][1]["content"]
