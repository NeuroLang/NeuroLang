"""
The ``claude`` CLI that the Claude Code backend runs.

With ``auth: login`` the backend uses the sign-in the user made with
``claude`` themselves, on this machine: NeuroLang offers no login of its
own, it only reports the status of that sign-in (:func:`status`). The CLI
takes the credentials of the environment first, so ``ANTHROPIC_API_KEY``
and ``ANTHROPIC_AUTH_TOKEN`` must not be there: the Agent SDK can add
variables to the CLI's environment but not remove them, so :func:`settings`
makes a small script that unsets them and runs the CLI.
"""

import contextlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import ProviderConfig
from .providers import LLMError

# What the CLI reads before its own sign-in.
CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
STATUS_SECONDS = 30


@dataclass
class Status:
    """The sign-in of the CLI: whether, how (``claude.ai``, ``api_key``...)
    and as whom."""

    logged_in: bool
    method: str = ""
    email: str = ""


@dataclass
class CliSettings:
    """What runs the CLI: the *path* of its executable and the *env*
    added to its environment."""

    path: str
    env: dict = field(default_factory=dict)


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
        bool(data.get("loggedIn")), str(data.get("authMethod") or ""),
        str(data.get("email") or ""),
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


def _wrapper(folder: Path, executable_path: str, unset) -> str:
    """Write in *folder* a script running the CLI without the variables
    *unset*; return its path."""
    script = folder / "claude"
    script.write_text(
        "#!/bin/sh\n"
        + (f"unset {' '.join(unset)}\n" if unset else "")
        + f"exec {shlex.quote(executable_path)} \"$@\"\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@contextlib.contextmanager
def settings(provider: ProviderConfig):
    """
    Yield the :class:`CliSettings` running the CLI as *provider* says.

    ``auth: login`` needs a sign-in that is not an API key, and runs the
    CLI without the credentials of the environment. ``auth: key`` runs it
    with the key, ``ANTHROPIC_API_KEY``, and a configuration folder of its
    own, which nothing of the user's CLI is read from.
    """
    key = provider.api_key()  # refuses a login on a shared server
    if provider.auth == "login":
        require_login()
    real = executable()
    with tempfile.TemporaryDirectory(prefix="neurolang-claude-") as folder:
        folder = Path(folder)
        if provider.auth == "login":
            path = _wrapper(folder, real, CREDENTIALS)
            yield CliSettings(path)
            return
        config = folder / "config"
        config.mkdir()
        path = _wrapper(folder, real, ("ANTHROPIC_AUTH_TOKEN",))
        yield CliSettings(path, {
            "ANTHROPIC_API_KEY": key, "CLAUDE_CONFIG_DIR": str(config),
        })
