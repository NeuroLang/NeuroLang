"""
Sharing the assistant's answers with a feedback server.

When the ``feedback`` section of the LLM configuration is ``enabled``,
or the environment variable ``NEUROLANG_LLM_FEEDBACK_URL`` gives the
address of a server, each answer of the assistant is posted to that
:mod:`.feedback_server` (:meth:`FeedbackClient.share`), to be rated and
corrected there or with ``.rate`` and ``.refine`` in the REPL.

What is sent is :func:`answer_record`: the question, the query, its
explanation, the steps of the agent (their observations cut), the rows
of the query as the agent saw them, the engine's name, the provider's
name and model, and the optional user name. API keys, the providers'
addresses and headers, and the paths of the configuration and the data
are never sent; the home directory is written ``~`` in the texts.
"""

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import httpx

FEEDBACK_URL_ENV = "NEUROLANG_LLM_FEEDBACK_URL"
MAX_OBSERVATION = 1000


class FeedbackError(Exception):
    """Raised when the feedback server cannot be reached or refuses."""


def _client_version() -> str:
    try:
        return version("neurolang-apps")
    except PackageNotFoundError:
        return "unknown"


def _anonymous(value):
    """Return *value* with the home directory written ``~`` in its texts."""
    home = str(Path.home())
    if isinstance(value, str):
        return value.replace(home, "~") if len(home) > 1 else value
    if isinstance(value, dict):
        return {key: _anonymous(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_anonymous(item) for item in value]
    return value


def _rows(result) -> str:
    """Return the rows of the last run of the result's query, if any."""
    query = " ".join(result.query.split())
    for step in reversed(result.steps):
        ran = " ".join(str(step.arguments.get("query", "")).split())
        if step.action == "run_query" and ran == query:
            return step.observation
    return ""


def answer_record(
    question: str, result, provider, strategy: str, engine: str = "",
    user: str = "",
) -> dict:
    """Return what is shared of the agent's *result* for *question*."""
    return _anonymous({
        "client_version": _client_version(), "engine": engine,
        "provider": provider.name, "model": provider.model,
        "strategy": strategy, "language": result.language,
        "question": question, "query": result.query,
        "explanation": result.explanation,
        "steps": [
            {"thought": step.thought, "action": step.action,
             "arguments": step.arguments,
             "observation": step.observation[:MAX_OBSERVATION]}
            for step in result.steps
        ],
        "rows": _rows(result)[:MAX_OBSERVATION], "valid": result.valid,
        "error": result.error, "seconds": round(result.seconds, 3),
        "calls": result.calls,
        "tokens": result.usage.get("input", 0)
        + result.usage.get("output", 0),
        "user": user,
    })


class FeedbackClient:
    """
    The client of the feedback server at *url*, its ratings signed
    *user*. *say* prints a message for the user: that the answers are
    shared, once, and that sharing failed, once.
    """

    def __init__(
        self, url: str, user: str = "", say: Callable = print,
        transport=None,
    ):
        if transport is None and sys.platform == "emscripten":
            from .browser import BrowserTransport

            transport = BrowserTransport()
        self.url = url
        self.user = user
        self.say = say
        self._told = self._warned = False
        self._http = httpx.Client(
            base_url=url.rstrip("/"), timeout=10, transport=transport
        )

    def _post(self, path: str, data: dict) -> str:
        """Post *data* to *path* and return the id the server gives."""
        try:
            response = self._http.post(path, json=data)
            response.raise_for_status()
            return response.json()["id"]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise FeedbackError(
                f"The feedback server {self.url} failed: {exc}"
            ) from exc

    def share(self, question, result, provider, strategy, engine=""):
        """Post the answer and return its id, or None if it failed."""
        if not self._told:
            self._told = True
            self.say(
                f"The questions and the answers of the assistant are shared "
                f"with {self.url}, to be rated and to improve it (feedback "
                "in the LLM configuration)."
            )
        try:
            return self._post("/api/answers", answer_record(
                question, result, provider, strategy, engine, self.user
            ))
        except FeedbackError as exc:
            if not self._warned:
                self._warned = True
                self.say(f"{exc}. The answers are not shared.")
            return None

    def rate(
        self, answer_id: str, verdict: str, comment: str = "",
        corrected_query: str = "", corrected_rows: str = "",
    ) -> str:
        """Post a rating of the answer *answer_id*; raise FeedbackError."""
        data = {"verdict": verdict, "user": self.user, "comment": comment}
        if corrected_query:
            data.update(
                corrected_query=corrected_query,
                corrected_query_checked=True,
                corrected_rows=corrected_rows[:MAX_OBSERVATION],
            )
        return self._post(
            f"/api/answers/{answer_id}/feedback", _anonymous(data)
        )


def feedback_client(config, say: Callable = print, transport=None):
    """
    Return the client of the feedback server of *config*, an
    :class:`~.config.LLMConfig`, or None when sharing is off.
    """
    settings = config.feedback
    url = os.environ.get(FEEDBACK_URL_ENV) or (
        settings.url if settings.enabled else ""
    )
    if not url:
        return None
    return FeedbackClient(url, settings.user, say, transport)
