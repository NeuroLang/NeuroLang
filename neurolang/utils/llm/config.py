"""
Configuration of the LLM query assistant.

The providers and the settings of the assistant are read from a YAML
file (see ``llm.example.yaml``, next to this module). It is looked up,
in order, at the path given explicitly (``--llm-config``), at the path
of the ``NEUROLANG_LLM_CONFIG`` environment variable, at
``~/.config/neurolang/llm.yaml`` and, failing those, the example file
shipped with NeuroLang is used.

A provider speaks the OpenAI chat completions API (``api: openai``:
OpenRouter, Albert, vLLM, Ollama…), the Anthropic messages API
(``api: anthropic``) or is a local command reading the conversation on
its standard input (``api: command``, e.g. ``claude -p``). Its API key
is read from the environment variable named by ``api_key_env``: keys
are never written in the file.
"""

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml

CONFIG_ENV = "NEUROLANG_LLM_CONFIG"
USER_CONFIG = Path("~/.config/neurolang/llm.yaml")
EXAMPLE_CONFIG = Path(__file__).with_name("llm.example.yaml")

APIS = ("openai", "anthropic", "command")
TOOL_CALLING = ("text", "native")
STRATEGIES = ("react", "icl")
LANGUAGES = ("auto", "squall", "datalog")


class LLMConfigError(ValueError):
    """Raised when the LLM configuration is invalid or incomplete."""


@dataclass(frozen=True)
class ProviderConfig:
    """
    An LLM provider: its API, endpoint, model and credentials.

    *api* is ``openai`` or ``anthropic``, the protocol spoken at
    *base_url*, or ``command``: *command* is then run for each request,
    reading the conversation on its standard input and writing the
    reply on its standard output; ``{system}`` in its arguments is
    replaced by the system prompt. The API key is read from the
    environment variable *api_key_env*, if any (local servers need
    none). *tool_calling* tells
    whether the ReAct agent uses the provider's ``native`` tool calls or
    the ``text`` Thought/Action/Observation protocol, which any chat
    model can follow. *requests_per_minute* spaces the requests to the
    provider's rate limit, if any.
    """

    name: str
    api: str
    model: str
    base_url: str = ""
    api_key_env: Optional[str] = None
    headers: dict = field(default_factory=dict)
    tool_calling: str = "text"
    max_tokens: int = 2048
    command: tuple = ()
    requests_per_minute: Optional[float] = None

    def api_key(self) -> Optional[str]:
        """Return the API key of the provider, read from the environment."""
        if not self.api_key_env:
            return None
        key = os.environ.get(self.api_key_env)
        if not key:
            raise LLMConfigError(
                f"The LLM provider {self.name!r} needs an API key: set the "
                f"environment variable {self.api_key_env}."
            )
        return key


@dataclass
class AssistantConfig:
    """
    Settings of the query assistant.

    *strategy* is ``react`` (an agent exploring the engine with tools)
    or ``icl`` (a few-shot prompt, then validation and repair).
    *language* is ``auto`` (the language of the REPL), ``squall`` or
    ``datalog``. *samples* above 1 generate several ICL candidates and
    keep the one most of them agree on (self-consistency).
    *query_timeout* and *query_memory_mb* stop a query the agent runs
    after that many seconds, or when it uses that much memory: it runs in
    a child process. With *comment_rules*, the accepted query is shown
    with a comment line explaining each rule (see
    :mod:`~neurolang.utils.llm.comments`).
    """

    strategy: str = "react"
    language: str = "auto"
    max_steps: int = 8
    max_repairs: int = 3
    few_shot_k: int = 6
    samples: int = 1
    temperature: float = 0.0
    timeout: float = 60.0
    preview_rows: int = 10
    query_timeout: float = 30.0
    query_memory_mb: float = 4096.0
    comment_rules: bool = True


@dataclass
class LLMConfig:
    """The providers and the assistant settings of a configuration file."""

    providers: dict
    default_provider: str
    assistant: AssistantConfig = field(default_factory=AssistantConfig)
    path: Optional[Path] = None

    def provider(self, name: Optional[str] = None) -> ProviderConfig:
        """Return the provider *name*, by default the default provider."""
        name = name or self.default_provider
        if name not in self.providers:
            available = ", ".join(sorted(self.providers)) or "none"
            raise LLMConfigError(
                f"Unknown LLM provider {name!r}. Available providers: "
                f"{available}."
            )
        return self.providers[name]


def config_path(path=None) -> Path:
    """Return the path of the configuration file to read (see the module)."""
    if path:
        return Path(path).expanduser()
    if os.environ.get(CONFIG_ENV):
        return Path(os.environ[CONFIG_ENV]).expanduser()
    user = USER_CONFIG.expanduser()
    if user.exists():
        return user
    return EXAMPLE_CONFIG


def _check_choice(value, choices, what: str):
    """Return *value*, raising LLMConfigError if it is not in *choices*."""
    if value not in choices:
        raise LLMConfigError(
            f"Invalid {what} {value!r}: expected one of {', '.join(choices)}."
        )
    return value


def _provider(name: str, entry) -> ProviderConfig:
    """Return the provider *name* described by the YAML mapping *entry*."""
    if not isinstance(entry, dict):
        raise LLMConfigError(f"Provider {name!r} must be a mapping.")
    if "api_key" in entry:
        raise LLMConfigError(
            f"Provider {name!r} has an inline api_key: put the key in an "
            "environment variable and name it with api_key_env."
        )
    required = ("command",) if entry.get("api") == "command" else (
        "base_url", "model"
    )
    missing = [key for key in required if not entry.get(key)]
    if missing:
        raise LLMConfigError(
            f"Provider {name!r} lacks {' and '.join(missing)}."
        )
    known = {f.name for f in fields(ProviderConfig)} - {"name"}
    unknown = sorted(set(entry) - known)
    if unknown:
        raise LLMConfigError(
            f"Provider {name!r} has unknown keys: {', '.join(unknown)}."
        )
    entry = {"api": "openai", "model": "", **entry}
    if entry.get("command"):
        entry["command"] = tuple(str(arg) for arg in entry["command"])
    provider = ProviderConfig(name=name, **entry)
    _check_choice(provider.api, APIS, f"api of provider {name!r}")
    _check_choice(
        provider.tool_calling, TOOL_CALLING,
        f"tool_calling of provider {name!r}",
    )
    if provider.api == "command" and provider.tool_calling != "text":
        raise LLMConfigError(
            f"Provider {name!r} is a command: it supports only "
            "tool_calling: text."
        )
    return provider


def _assistant(entry) -> AssistantConfig:
    """Return the assistant settings of the YAML mapping *entry*."""
    entry = entry or {}
    known = {f.name for f in fields(AssistantConfig)}
    unknown = sorted(set(entry) - known)
    if unknown:
        raise LLMConfigError(
            f"Unknown assistant settings: {', '.join(unknown)}."
        )
    assistant = AssistantConfig(**entry)
    _check_choice(assistant.strategy, STRATEGIES, "strategy")
    _check_choice(assistant.language, LANGUAGES, "language")
    return assistant


def parse_config(data: dict, path: Optional[Path] = None) -> LLMConfig:
    """Return the configuration described by the YAML mapping *data*."""
    if not isinstance(data, dict):
        raise LLMConfigError("The LLM configuration must be a mapping.")
    providers = {
        name: _provider(name, entry)
        for name, entry in (data.get("providers") or {}).items()
    }
    if not providers:
        raise LLMConfigError("The LLM configuration declares no provider.")
    default = data.get("default_provider") or next(iter(providers))
    config = LLMConfig(
        providers, default, _assistant(data.get("assistant")), path
    )
    config.provider()
    return config


def load_config(path=None) -> LLMConfig:
    """Read the configuration file (see :func:`config_path`)."""
    path = config_path(path)
    try:
        with open(path, encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except OSError as exc:
        raise LLMConfigError(
            f"Cannot read the LLM configuration {path}: {exc.strerror}."
        ) from exc
    except yaml.YAMLError as exc:
        raise LLMConfigError(
            f"Invalid YAML in the LLM configuration {path}: {exc}"
        ) from exc
    return parse_config(data, path)
