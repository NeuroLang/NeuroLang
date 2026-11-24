"""
The ``claude`` CLI that the Claude Code backend runs.

The backend uses the sign-in the user made with ``claude`` themselves, on
this machine: NeuroLang offers no login of its own, it only checks that
sign-in (:func:`require_login`). The CLI takes the credentials of the
environment first, so ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_AUTH_TOKEN``
must not be there: the Agent SDK can add variables to the CLI's
environment but not remove them, so :func:`settings` makes a small script
that unsets them and runs the CLI.
"""

import contextlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .providers import LLMError

# What the CLI reads before its own sign-in.
CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
STATUS_SECONDS = 30


@dataclass
class Status:
    """The sign-in of the CLI: whether, and how (``claude.ai``,
    ``api_key``...)."""

    logged_in: bool
    method: str = ""


def executable() -> str:
    """Return the path of the ``claude`` CLI."""
    path = shutil.which("claude")
    if path is None:
        raise LLMError(
            "the claude CLI was not found: install Claude Code "
            "(https://claude.com/claude-code) and sign in with `claude`."
        )
    return path


def clean_env() -> dict:
    """Return the environment without the credentials the CLI reads."""
    return {k: v for k, v in os.environ.items() if k not in CREDENTIALS}


def status() -> Status:
    """Return the sign-in of the CLI, as it is without any key."""
    try:
        completed = subprocess.run(
            [executable(), "auth", "status", "--json"], env=clean_env(),
            capture_output=True, text=True, timeout=STATUS_SECONDS,
        )
        data = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise LLMError(f"cannot read the sign-in of claude: {exc}") from exc
    return Status(
        bool(data.get("loggedIn")), str(data.get("authMethod") or "")
    )


def require_login() -> Status:
    """Return the sign-in of the CLI, raising LLMError unless it has one
    that is not an API key."""
    current = status()
    if not current.logged_in or current.method == "api_key":
        raise LLMError(
            "not signed in to Claude: run `claude` in a terminal to sign "
            "in, or use a provider with auth: key."
        )
    return current


@contextlib.contextmanager
def settings(provider):
    """
    Yield the path of a ``claude`` that runs the CLI without the
    credentials of the environment, once the sign-in is checked and the
    use of *provider* allowed (``api_key`` refuses a shared server).
    """
    provider.api_key()
    require_login()
    real = executable()
    with tempfile.TemporaryDirectory(prefix="neurolang-claude-") as folder:
        script = Path(folder) / "claude"
        script.write_text(
            "#!/bin/sh\n"
            f"unset {' '.join(CREDENTIALS)}\n"
            f"exec {shlex.quote(real)} \"$@\"\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        yield str(script)
