"""
The REPL of the browser's page, run by this machine's Python.

The page (``doc/_static/repl``) runs NeuroLang in its worker, on Pyodide,
within the memory a tab is given. When its ``bundle.json`` says
``"backend": "local"``, as ``neurolang-repl-serve --local`` writes it, the
page starts ``local_worker.js`` instead, which sends what the page asks to
the server, and so to a :class:`LocalBackend`: the engine is built by this
machine's Python from its data directory, with all of its analyses, and the
REPL is the page's own, :class:`~.web.WebRepl`, whose replies the page
shows as it does those of Pyodide.

What the REPL tells while it works (the steps of a build, of the AI
assistant) goes to every page listening (:meth:`LocalBackend.listen`), the
server sending it as events.
"""

import json
import os
import queue
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

from neurolang.confine import lifted
from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.tui import web
from neurolang_apps.utils.tui.interrupt import _raise_in_thread


def reply_payload(reply: web.Reply) -> dict:
    """Return *reply* as the page takes it, as ``worker.js`` gives it."""
    return {
        "html": reply.html, "viewers": reply.viewers,
        "failed": reply.failed, "tables": reply.tables,
        "prompt": reply.prompt, "mode": reply.mode,
        "predicates": reply.predicates,
        "ratable": getattr(reply, "ratable", False),
        "graph": getattr(reply, "graph", ""),
        "tokens": getattr(reply, "tokens", {}),
    }


# The variables of a session's process (neurolang-repl-serve --users) telling
# its user's assistant, as JSON (provider, model, base_url, key), and, "0",
# that they may not give their own key (tools/serve_repl has them too).
ASSISTANT_ENV = "NEUROLANG_REPL_LLM"
OWN_KEY_ENV = "NEUROLANG_REPL_OWN_KEY"
# "1": the user brings their own key (a guest): the server's is not theirs.
BRING_KEY_ENV = "NEUROLANG_REPL_BRING_KEY"
OWN_KEY_REFUSED = ("Your assistant's key is the one your admin set: you "
                   "cannot give your own.")

# What the page says of a rating stored on this server.
RATING_NOTE = (
    "A rating stores the query with its first rows or its error, and for an "
    "answer of the assistant the question and its steps, on this server, "
    "with your user name and groups."
)


def apply_env(env: dict) -> None:
    """Put the variables *env* in this process's environment; one whose
    value is None is taken out (a key of the server, from a guest's)."""
    for name, value in env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class LocalBackend:
    """
    The engine and the REPL of the page, in this process.

    The engines are built from *data_dir* at *resolution*, with all their
    analyses unless *light* (those a tab can hold, as in the browser).
    *feedback*, ``(url, doc)``, is where a rated answer goes (see
    :meth:`~.web.WebRepl.set_feedback`), or ``{"path": ..., "user": ...,
    "groups": ...}`` the database of this machine where it is stored,
    signed (see :meth:`~.web.WebRepl.set_feedback_store`). *build*,
    ``build(name, say)``, builds an engine, telling its steps with ``say``:
    by default
    :func:`~neurolang_apps.utils.engine_registry.build_engine`.
    """

    def __init__(
        self, data_dir, resolution: float = 2.0, light: bool = False,
        feedback: Optional[tuple] = None, build: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.resolution = resolution
        self.light = light
        self.feedback = feedback
        self._build = build or self._build_engine
        self.repl: Optional[web.WebRepl] = None
        self._page_key = False  # whether the page gave the assistant's key
        # The values the variables of the user's identity had before it
        # (see use_identity).
        self._before_identity = {}
        # One request at a time works on the REPL; the thread of the one
        # that does, for a stop to interrupt it.
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._listeners = []
        self._listeners_lock = threading.Lock()

    # -- What the REPL tells while it works -------------------------------

    def listen(self) -> queue.Queue:
        """Return a queue of the messages to come, as JSON texts."""
        listener = queue.Queue()
        with self._listeners_lock:
            self._listeners.append(listener)
        return listener

    def forget(self, listener: queue.Queue) -> None:
        """Stop sending messages to *listener*."""
        with self._listeners_lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def emit(self, message: dict) -> None:
        """Send *message* to every page listening."""
        text = json.dumps(message)
        with self._listeners_lock:
            for listener in self._listeners:
                listener.put(text)

    def say(self, message: str) -> None:
        """Tell the pages a step of the work, as their status."""
        self.emit({"type": "status", "message": message + "…"})

    # -- The requests of the page -----------------------------------------

    def _build_engine(self, name: str, say: Callable):
        yaml_path = None
        if self.light:
            yaml_path = web.lightweight_engines_yaml(tempfile.mkdtemp())
        # The files an engine reads by a relative path (the atlas of
        # tractography) are in its data directory, which its downloads make
        # on a first start.
        self.data_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.data_dir)
        return engine_registry.build_engine(
            name, self.data_dir, self.resolution, yaml_path=yaml_path,
            progress=say,
        )

    def _working(self, work: Callable):
        """Return *work()*, one request at a time, which a stop interrupts."""
        with self._lock:
            self._worker = threading.current_thread()
            try:
                return work()
            finally:
                self._worker = None

    def stop(self) -> bool:
        """Interrupt the request at work, if any; return whether there was
        one."""
        worker = self._worker
        if worker is None:
            return False
        _raise_in_thread(worker, KeyboardInterrupt)
        return True

    def start(self, engine: str) -> dict:
        """
        Build the engine *engine* and a REPL on it, and return what the page
        is told when it is ready. Each start builds them anew, as the page
        does when it restarts: the rules of a session are not kept.
        """
        return self._working(lambda: self._start(engine))

    def _start(self, engine: str) -> dict:
        with lifted():  # the engine's commands are the server's
            nl = self._build(engine, self.say)
        self.say("Starting the REPL")
        options = {}
        # The metadata of the studies of Neurosynth, whose ids are
        # PubMed's, let the page link the studies of a result.
        if engine.startswith("neurosynth"):
            try:
                from neurolang.frontend.neurosynth_utils import (
                    fetch_study_metadata,
                )
                options["study_metadata"] = fetch_study_metadata(
                    self.data_dir, verbose=0
                )
            except Exception:
                pass
        self._page_key = False
        repl = web.WebRepl(nl, engine, self.resolution, **options)
        repl.set_event_sink(lambda text: self.emit(json.loads(text)))
        self._use_machine_llm(repl)
        can_rate = False
        if isinstance(self.feedback, dict):  # this machine's database
            repl.set_feedback_store(**self.feedback)
            can_rate = repl.can_rate()
        elif self.feedback:
            repl.set_feedback(*self.feedback)
            can_rate = repl.can_rate()
        self.repl = repl
        from neurolang_apps.utils.llm.browser import providers_for_form
        import neurolang
        return {
            "banner": repl.banner(),
            "source": f"the Python of this machine ({sys.executable}), "
            f"NeuroLang from {Path(neurolang.__file__).parent.parent}",
            "providers": providers_for_form(),
            "examples": repl.examples(),
            "questions": repl.questions(),
            "predicates": repl.predicates(),
            "commands": repl.commands(),
            "canRate": can_rate,
            # whether the page may give a key of its own
            "ownKey": os.environ.get(OWN_KEY_ENV) != "0",
            # where a rating goes, when it stays on this server
            **({"ratingNote": RATING_NOTE}
               if isinstance(self.feedback, dict) else {}),
            # whether the assistant of this machine can answer, for the
            # page to suggest a question (a version of the page that
            # predates it leaves it out)
            "assistant": getattr(repl, "assistant_ready", lambda: False)(),
        }

    def _repl(self) -> web.WebRepl:
        if self.repl is None:
            raise RuntimeError("The engine is not started")
        return self.repl

    def submit(self, text: str) -> dict:
        """Run *text* in the REPL and return its reply for the page."""
        def work():
            try:
                payload = reply_payload(self._repl().submit(text))
                if payload["tokens"]:  # whose key: the page's, or the server's
                    payload["tokens"]["key"] = \
                        "page" if self._page_key else "server"
                return payload
            except KeyboardInterrupt:
                return {
                    "html": "<pre>Stopped.</pre>", "failed": True,
                    "viewers": [], "tables": [],
                }
        return self._working(work)

    def set_llm(
        self, provider: str = "", key: str = "", model: str = "",
        baseUrl: str = "",
    ) -> None:
        """Use the AI assistant of *provider* with *key*, or that of this
        machine without a key; a user who may not give their own key
        cannot."""
        repl = self._repl()
        if not key:
            self._page_key = False
            self._use_machine_llm(repl)
        elif os.environ.get(OWN_KEY_ENV) == "0":
            raise ValueError(OWN_KEY_REFUSED)
        else:
            repl.set_llm(provider, key, model, baseUrl)
            self._page_key = True
        self._tell_assistant()

    def use_identity(self, env: dict, groups=None) -> dict:
        """
        Take the variables *env* of the user as they are now (their
        assistant, whether they may give their own key, the keys of the
        server's configuration), which an admin or the user changed while
        the session runs, and their *groups*, which sign their ratings, if
        given. A variable of the identity before that *env* lacks is as it
        was before it (a key revoked is taken out). If they changed, the
        assistant is theirs at once, unless the page gives a key that they
        may still give. Return what the page is told.
        """
        before = dict(os.environ)
        apply_env(self._before_identity)
        self._before_identity = {name: os.environ.get(name) for name in env}
        apply_env(env)
        if groups is not None and isinstance(self.feedback, dict) and \
                groups != self.feedback.get("groups"):
            self.feedback = {**self.feedback, "groups": groups}
            if self.repl is not None:
                self.repl.set_feedback_store(**self.feedback)
        if self.repl is not None and dict(os.environ) != before and (
            not self._page_key or os.environ.get(OWN_KEY_ENV) == "0"
        ):
            self._page_key = False
            self._use_machine_llm(self.repl)
        return self._tell_assistant()

    def _tell_assistant(self) -> dict:
        """Tell the pages whether the assistant can answer and whether the
        user may give their own key; return it."""
        state = {
            "type": "assistant",
            "ready": bool(self.repl and self.repl.assistant_ready()),
            "ownKey": os.environ.get(OWN_KEY_ENV) != "0",
        }
        self.emit(state)
        return state

    @staticmethod
    def _use_machine_llm(repl: web.WebRepl) -> None:
        """
        Have the assistant of *repl* use the configuration of this machine,
        as ``neurolang-query`` does (its file, and the keys of the
        environment), until the page gives a key; without one, the
        assistant asks the page for a key, as in the browser.
        """
        if os.environ.get(BRING_KEY_ENV) == "1":  # theirs, from the page
            repl.set_llm("anthropic", "")
            return
        assistant = os.environ.get(ASSISTANT_ENV)
        if assistant:  # that of the user, or of their group
            settings = json.loads(assistant)
            repl.set_llm(settings["provider"], settings.get("key", ""),
                         settings.get("model", ""),
                         settings.get("base_url", ""))
            return
        state = repl.app.assistant
        state.provider = None
        try:
            state.reload()
        except Exception:
            repl.set_llm("anthropic", "")

    def rate(self, verdict: str, comment: str = "") -> dict:
        """Share the last answer of the assistant with its rating."""
        return self._repl().rate_answer(verdict, comment)

    def upload(self, name: str, data: bytes) -> dict:
        """Keep the file the page uploaded (WebRepl.upload)."""
        return self._working(lambda: self._repl().upload(name, data))

    def download(self, block: int, kind: str, value: str = "") -> web.Download:
        """Return the file of a result for the page to save."""
        return self._working(
            lambda: self._repl().download(block, kind, value)
        )
