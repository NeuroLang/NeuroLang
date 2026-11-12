"""
Sharing the assistant's answers with a feedback server.

When the ``feedback`` section of the LLM configuration is ``enabled``,
or the environment variable ``NEUROLANG_LLM_FEEDBACK_URL`` gives the
address of a server, each answer of the assistant is posted to that
:mod:`.feedback_server` (:meth:`FeedbackClient.share`), to be rated and
corrected there or with ``.rate`` and ``.refine`` in the REPL. With
``grist_doc``, the answers are instead rows of the table ``Answers`` of
that Grist document (:class:`GristFeedbackClient`).

What is sent is :func:`answer_record`: the question, the query, its
explanation, the steps of the agent (their observations cut), the rows
of the query as the agent saw them, the engine's name, the provider's
name and model, and the optional user name. API keys, the providers'
addresses and headers, and the paths of the configuration and the data
are never sent; the home directory is written ``~`` in the texts.
"""

import json
import os
import sys
import time
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


def query_record(
    query: str, rows: str, error: str, engine: str = "", user: str = "",
) -> dict:
    """
    Return what is shared of a query run without the assistant: no
    question and no steps, its *rows* (a preview) or its *error*.
    """
    return _anonymous({
        "client_version": _client_version(), "engine": engine,
        "question": "", "query": query, "steps": [],
        "rows": rows[:MAX_OBSERVATION], "valid": not error,
        "error": error[:MAX_OBSERVATION], "user": user,
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

    def _headers(self) -> dict:
        return {}

    def _send(self, method, path, data, read=lambda reply: reply["id"]):
        """Send *data* to *path* and return what *read* takes of the reply."""
        try:
            response = self._http.request(
                method, path, json=data, headers=self._headers()
            )
            response.raise_for_status()
            return read(response.json())
        except (httpx.HTTPError, ValueError, LookupError) as exc:
            raise FeedbackError(
                f"The feedback server {self.url} failed: {exc}"
            ) from exc

    def _share(self, record: dict) -> str:
        return self._send("POST", "/api/answers", record)

    def _rate(self, answer_id: str, data: dict) -> str:
        return self._send("POST", f"/api/answers/{answer_id}/feedback", data)

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
            return self._share(answer_record(
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
        return self._rate(answer_id, _anonymous(data))


class GristFeedbackClient(FeedbackClient):
    """
    The client of the Grist document *doc* at *url*: each answer is a row
    of its table ``Answers``, whose id is the answer's; a rating sets the
    row's ``verdict`` (good, bad or unsure), ``comment`` and
    ``corrected_query``. The API key is in the environment variable
    *api_key_env*. Without one (``api_key_env=""``) the client is
    anonymous, as a web page must be (Grist does not accept its key from
    another origin): the document lets everyone add rows, and
    :meth:`share_rated` adds the answer with its verdict, in one row.
    """

    VERDICTS = {"correct": "good", "incorrect": "bad"}

    def __init__(
        self, url: str, doc: str, api_key_env: str = "GRIST_API_KEY",
        user: str = "", say: Callable = print, transport=None,
    ):
        super().__init__(url, user, say, transport)
        self.url = f"{self.url.rstrip('/')}/{doc}"
        self.records = f"/api/docs/{doc}/tables/Answers/records"
        self.api_key_env = api_key_env

    def _headers(self) -> dict:
        if not self.api_key_env:
            return {}
        key = os.environ.get(self.api_key_env)
        if not key:
            raise FeedbackError(
                f"{self.api_key_env} is not set: no API key for the Grist "
                f"document {self.url}"
            )
        return {"Authorization": f"Bearer {key}"}

    def _share(self, record: dict) -> str:
        record.update(steps=json.dumps(record["steps"]), time=time.time())
        return self._send(
            "POST", self.records, {"records": [{"fields": record}]},
            lambda reply: str(reply["records"][0]["id"]),
        )

    def share_rated(
        self, question, result, provider, strategy, engine, verdict,
        comment="",
    ) -> str:
        """
        Add the answer with its *verdict* and *comment* as a row and return
        its id; raise FeedbackError. Nothing of an answer is sent but when
        it is rated.
        """
        return self.share_record(
            answer_record(
                question, result, provider, strategy, engine, self.user
            ),
            verdict, comment,
        )

    def share_record(self, record: dict, verdict: str, comment="") -> str:
        """
        Add *record* (:func:`answer_record`, :func:`query_record`) with its
        *verdict* and *comment* as a row and return its id; raise
        FeedbackError.
        """
        record.update(
            verdict=self.VERDICTS.get(verdict, verdict),
            comment=_anonymous(comment),
        )
        return self._share(record)

    def _rate(self, answer_id: str, data: dict) -> str:
        if not str(answer_id).isdigit():
            raise FeedbackError(f"{answer_id} is not a row of {self.url}")
        fields = {
            "verdict": self.VERDICTS.get(data["verdict"], data["verdict"]),
            "comment": data["comment"],
        }
        if data.get("corrected_query"):
            fields["corrected_query"] = data["corrected_query"]
        return self._send(
            "PATCH", self.records,
            {"records": [{"id": int(answer_id), "fields": fields}]},
            lambda reply: answer_id,
        )


class StoreFeedbackClient:
    """
    Rated answers written in the feedback database *path* of this machine
    (a :class:`~neurolang_apps.utils.llm.feedback_server.FeedbackStore`,
    which several processes may write at once), signed by the *user* and
    their *groups*, as :class:`GristFeedbackClient` shares them.
    """

    VERDICTS = {"good": "correct", "bad": "incorrect"}

    def __init__(self, path, user: str = "", groups: str = ""):
        from .feedback_server import FeedbackStore

        self.store = FeedbackStore(path)
        self.user = user
        self.groups = groups

    def share_rated(
        self, question, result, provider, strategy, engine, verdict,
        comment: str = "",
    ) -> str:
        """Store the answer of the assistant with its *verdict*."""
        return self.share_record(
            answer_record(question, result, provider, strategy, engine),
            verdict, comment,
        )

    def share_record(self, record: dict, verdict: str, comment="") -> str:
        """Store the answer or the query *record* with its *verdict* and
        *comment*, signed; return its id; raise :class:`FeedbackError`."""
        signed = {"user": self.user, "groups": self.groups}
        try:
            answer = self.store.add_answer(
                {**record, **signed}, question=bool(record.get("question"))
            )
            self.store.add_feedback(answer, {
                "verdict": self.VERDICTS.get(verdict, verdict),
                "comment": comment, **signed,
            })
        except Exception as error:
            raise FeedbackError(f"The rating is not stored: {error}")
        return answer


def feedback_client(config, say: Callable = print, transport=None):
    """
    Return the client of the feedback server of *config*, an
    :class:`~.config.LLMConfig`, or None when sharing is off.
    """
    settings = config.feedback
    if settings.enabled and settings.grist_doc:
        return GristFeedbackClient(
            settings.url, settings.grist_doc, settings.api_key_env,
            settings.user, say, transport,
        )
    url = os.environ.get(FEEDBACK_URL_ENV) or (
        settings.url if settings.enabled else ""
    )
    if not url:
        return None
    return FeedbackClient(url, settings.user, say, transport)
