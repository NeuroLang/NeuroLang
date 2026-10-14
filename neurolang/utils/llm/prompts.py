"""
The prompts of the LLM query assistant.

The system prompt tells the model the target language with a short
reference of its syntax, the predicates of the engine and the examples
most similar to the question. The ReAct agent's tools are described
once (:data:`TOOLS`), sent to the provider as native tools or written
in the prompt for the text protocol.
"""

import json

LANGUAGE_NAMES = {"squall": "SQUALL", "datalog": "Datalog"}

SQUALL_REFERENCE = """\
SQUALL is a controlled English for NeuroLang. A program is a sequence
of sentences, each ending with a full stop; the last one is a query
starting with `obtain`. Rules start with `define as Name`.
- Nouns are relations, capitalised or not: `Study` is `study`. A noun
  over several columns takes a tuple of variables separated by `;`:
  `every Peak_reported (?i; ?j; ?k; ?s)`; `_` is a column left out.
- Variables are written `?name`. Strings are in single quotes: 'pain'.
- Determiners: `every` (all), `a` (there exists), `no`, and `the`
  referring back to a noun introduced before.
- A transitive verb is a binary relation, subject first:
  `every Study that mentions 'pain'` is study(s), mentions(s, 'pain').
  `~verb` swaps its arguments. Verbs are not inflected: write the
  relation's name, e.g. `does not mentions 'pain'`.
- Relative clauses: `that`, `such that`, `where`; combine with `and`,
  `or`, `not`/`does not`. `?x is 'value'` binds a variable.
- Comparisons: `?w is greater than 0.1`, `greater equal than`,
  `lower than`, `lower equal than`, `equal to`, `not equal to`.
- A call of a predicate or function: `where startswith('lh', ?n) holds`;
  arithmetic: `?bf is (?a / ?b)`.
- Several variables in a rule head: `define as Pair for every Region
  and for every Term where a Study activates the Region and mentions
  the Term.`
- Aggregation: `obtain every Count of the Study.`, per group
  `define as N for every Term ; where every Count of the Study that
  mentions the Term.` (count, sum, max, min, average).
- Probabilities: `define as P with inferred probability every X.`,
  conditional `define as P with probability every A (?i; ?j; ?k; _)
  conditioned to every B (?s; ?t; _) such that ?t is 'auditory'.`
- Several obtain sentences give several results; `obtain ... as Name.`
  names one.
"""

DATALOG_REFERENCE = """\
NeuroLang Datalog: a program is a sequence of rules `head :- body`, one
per line; the query is the rule whose head is `ans(...)`, and there is
only one.
- Variables are lower-case identifiers; strings are in single quotes
  ('pain') and hold only letters, digits and spaces; numbers are
  written as is.
- The body is a conjunction separated by `,`: `ans(s) :- study(s),
  mentions(s, 'pain')`. Disjunction: several rules with the same head.
- Negation: `~mentions(s, 'pain')`; its variables must be bound by a
  positive atom of the body.
- Comparisons and arithmetic in parentheses: `(w > 0.1)`, `(c != 0)`,
  `(bf == (a / b))` binds bf.
- Functions are called as predicates: `startswith('lh', n)`.
- Aggregation, in a rule of its own that the query then reads:
  `term_count(t, n) :- AGGREGATE[t](mentions(s, t) @ count(s)) = n` and
  `ans(t, n) :- term_count(t, n)`; without group
  `AGGREGATE[](study(s) @ count(s)) = n` (count, sum, max, min, mean).
- Probabilities over probabilistic relations (e.g. a probabilistic
  choice): `ans(t, p) :- PROB[mentioned_term(t)] = p`, conditional
  `PROB[a(x) // b(x)] = p`.
"""

REFERENCES = {"squall": SQUALL_REFERENCE, "datalog": DATALOG_REFERENCE}

SCHEMA_NOTE = """\
Each predicate is listed with its columns, in order. An entity spanning
several columns is written sort(columns): reports(study(study_id),
voxel(i, j, k)) has the four columns study_id, i, j, k, used as
reports(s, i, j, k) in Datalog and as the verb `reports` between a Study
and a Voxel in SQUALL."""

ROLE = """\
You translate questions on neuroimaging data into queries of NeuroLang,
a probabilistic logic programming language. Answer in {language} only,
use only the predicates listed below, with their exact names and number
of columns, and write string constants as the engine stores them."""

ICL_INSTRUCTIONS = """\
Answer with the {language} program in a fenced code block (```{fence}),
followed by one sentence explaining what it computes. If the question
cannot be answered with these predicates, say so instead of writing a
query."""

REACT_INSTRUCTIONS = """\
Use the tools to look at the predicates and their values, check your
query and run it before answering: check that constants such as terms
exist with search_values, and that the query gives the expected kind of
result with run_query. Give the query with final_answer, once it passes
check_query."""

TEXT_PROTOCOL = """\
You can call these tools:
{tools}

To call a tool, write exactly:
Thought: <your reasoning, briefly>
Action: <the tool's name>
Action Input: <a JSON object with its arguments>
and stop there: its result is given back to you as `Observation: ...`.
Call one tool at a time. To answer, call final_answer, e.g.
Action: final_answer
Action Input: {{"query": "...", "explanation": "..."}}"""

TOOLS = [
    {
        "name": "list_predicates",
        "description": "List the predicates of the engine with their "
        "columns, description and size.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "describe_predicate",
        "description": "Describe a predicate and show a few of its rows.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "search_values",
        "description": "Find the values of a column of a predicate that "
        "contain a word, ignoring case, or the closest ones: use it to "
        "write constants as they are stored.",
        "parameters": {
            "type": "object",
            "properties": {
                "predicate": {"type": "string"},
                "column": {"type": "string"},
                "pattern": {"type": "string"},
            },
            "required": ["predicate", "column", "pattern"],
        },
    },
    {
        "name": "check_query",
        "description": "Parse the query and check it against the engine "
        "without running it; returns the error to correct, if any.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "run_query",
        "description": "Run the query and show its number of rows and its "
        "first rows.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "final_answer",
        "description": "Give the final query and a one-sentence "
        "explanation of what it computes.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["query", "explanation"],
        },
    },
]


def tool_lines(tools=TOOLS) -> str:
    """Return the description of *tools* for the text protocol."""
    lines = []
    for tool in tools:
        arguments = ", ".join(tool["parameters"].get("properties", {}))
        lines.append(f"- {tool['name']}({arguments}): {tool['description']}")
    return "\n".join(lines)


def format_examples(examples, language: str) -> str:
    """Return the *examples* as question and query pairs."""
    blocks = []
    for example in examples:
        query = example.query(language)
        if example.question:
            blocks.append(f"Question: {example.question}\n{query}")
        else:
            blocks.append(query)
    return "\n\n".join(blocks)


def system_prompt(
    language: str, schema: str, examples, strategy: str,
    text_tools: bool = False, engine_description: str = "",
) -> str:
    """
    Return the system prompt of the assistant.

    *strategy* is ``icl`` or ``react``; *text_tools* adds the text
    protocol of the tools, for providers without native tool calls.
    """
    name = LANGUAGE_NAMES[language]
    parts = [ROLE.format(language=name)]
    if engine_description:
        parts.append(f"The engine: {' '.join(engine_description.split())}")
    parts.append(f"# {name} syntax\n{REFERENCES[language]}")
    parts.append(f"# Predicates of the engine\n{SCHEMA_NOTE}\n{schema}")
    if examples:
        parts.append(
            f"# Examples\n{format_examples(examples, language)}"
        )
    if strategy == "icl":
        parts.append(ICL_INSTRUCTIONS.format(language=name, fence=language))
    else:
        parts.append(REACT_INSTRUCTIONS)
        if text_tools:
            parts.append(TEXT_PROTOCOL.format(tools=tool_lines()))
    return "\n\n".join(parts)


def repair_prompt(query: str, error: str, language: str) -> str:
    """Return the message asking the model to correct its *query*."""
    return (
        f"This {LANGUAGE_NAMES[language]} query fails:\n{query}\n\n"
        f"Error:\n{error}\n\nCorrect it, and answer as before."
    )


def observation(result: str) -> str:
    """Return the *result* of a tool as a text protocol observation."""
    return f"Observation: {result}"


def format_arguments(arguments: dict) -> str:
    """Return the arguments of a tool call as compact JSON."""
    return json.dumps(arguments, ensure_ascii=False)
