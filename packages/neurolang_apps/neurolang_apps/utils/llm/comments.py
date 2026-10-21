"""
A comment line explaining each rule of the assistant's query.

Once the agent has an accepted query, :func:`comment_rules` asks the
model, in one more request, for a line explaining each of its rules
(:func:`split_rules`), and :func:`commented_query` writes the query with
each line as a comment above its rule: ``% ...`` in Datalog, whose
parser ignores them, so that the commented query runs as it is. SQUALL
has no comments: its rules are shown with their comments, and the query
is kept without them. The model only writes the comments; the query is
the one that was checked.
"""

import json
import re

# The characters closing each bracket, to tell where a rule ends.
_CLOSING = {"(": ")", "[": "]", "{": "}"}
# A Datalog line ending so continues the rule on the next line.
_CONTINUED = re.compile(r"(:-|,|&|\||//|\N{LOGICAL AND})\s*$")
# The end of a SQUALL sentence: a dot not followed by a digit.
_SQUALL_END = re.compile(r"\.(?!\d)")
_STRING = re.compile(r"\"[^\"]*\"|'[^']*'")

# The longest comment kept, in characters.
MAX_COMMENT = 160

COMMENT_INSTRUCTIONS = """\
You explain NeuroLang queries to neuroscientists. The query below \
answers the question; its rules are numbered. Write, for each rule, one \
short line (at most 15 words) saying what it computes, in terms of the \
question, not of the syntax. Reply with a JSON list of exactly {count} \
strings, in the order of the rules, and nothing else."""


def _depth(text: str) -> int:
    """Return how many brackets of *text*, outside strings, are open."""
    depth = 0
    for char in _STRING.sub("", text):
        if char in _CLOSING:
            depth += 1
        elif char in _CLOSING.values():
            depth -= 1
    return depth


def _datalog_rules(query: str) -> list:
    """Return the rules of the Datalog *query*, a line or more each."""
    rules, current = [], []
    for line in query.splitlines():
        if not line.strip() or line.strip().startswith("%"):
            continue
        current.append(line.rstrip())
        text = "\n".join(current)
        if _depth(text) <= 0 and not _CONTINUED.search(text):
            rules.append(text)
            current = []
    if current:
        rules.append("\n".join(current))
    return rules


def _squall_rules(query: str) -> list:
    """Return the sentences of the SQUALL *query*, each ending its dot."""
    rules, start = [], 0
    masked = _STRING.sub(lambda m: "x" * len(m.group(0)), query)
    for end in _SQUALL_END.finditer(masked):
        rule = query[start:end.end()].strip()
        if rule:
            rules.append(rule)
        start = end.end()
    if query[start:].strip():
        rules.append(query[start:].strip())
    return rules


def split_rules(query: str, language: str) -> list:
    """
    Return the rules of *query*, in *language*, as they are written.

    A Datalog rule ends at the end of a line whose brackets are closed
    and which does not end with ``:-``, ``,``, ``&``, ``|`` or ``//``;
    its comment lines are left out. A SQUALL rule is a sentence.
    """
    if language == "squall":
        return _squall_rules(query)
    return _datalog_rules(query)


def comment_messages(question: str, rules: list) -> list:
    """Return the request asking for a comment on each of the *rules*."""
    numbered = "\n".join(
        f"{n}. {rule}" for n, rule in enumerate(rules, start=1)
    )
    return [
        {"role": "system", "content": COMMENT_INSTRUCTIONS.format(
            count=len(rules)
        )},
        {"role": "user", "content": f"Question: {question}\n\n{numbered}"},
    ]


def parse_comments(text: str, count: int) -> list:
    """
    Return the *count* comments of the model's reply *text*.

    The reply is a JSON list of strings, possibly in a code fence; each
    comment is kept to one line of at most :data:`MAX_COMMENT`
    characters. A :class:`ValueError` tells the reply is not such a
    list.
    """
    found = re.search(r"\[.*\]", text, re.S)
    if found is None:
        raise ValueError("The reply holds no JSON list.")
    comments = json.loads(found.group(0))
    if not (
        isinstance(comments, list) and len(comments) == count
        and all(isinstance(c, str) for c in comments)
    ):
        raise ValueError(f"The reply is not a list of {count} strings.")
    return [
        " ".join(c.replace("%", " ").split())[:MAX_COMMENT]
        for c in comments
    ]


def commented_query(rules: list, comments: list) -> str:
    """Return the *rules*, each under its comment line, ``% ...``."""
    return "\n".join(
        f"% {comment}\n{rule}" if comment else rule
        for rule, comment in zip(rules, comments)
    )


def comment_rules(complete, question: str, query: str, language: str):
    """
    Return the rules of *query* and a comment on each, from the model.

    *complete* sends a list of messages to the model and returns its
    reply. Return ``(rules, comments)``; a :class:`ValueError` tells the
    reply holds no comment per rule.
    """
    rules = split_rules(query, language)
    if not rules:
        raise ValueError("The query has no rule.")
    reply = complete(comment_messages(question, rules))
    return rules, parse_comments(reply.text, len(rules))
