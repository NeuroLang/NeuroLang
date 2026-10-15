"""
LLM query assistant: natural-language questions to NeuroLang queries.

The assistant turns a question into a SQUALL or Datalog query on an
engine, with an LLM reached through the providers of a YAML
configuration (:mod:`.config`): OpenRouter, Albert, Anthropic, any
OpenAI-compatible server, or a local command such as ``claude -p``. Two
strategies are offered (:mod:`.agent`): in-context learning with
validation and repair, and a ReAct agent exploring the engine with
tools. It needs the ``llm`` extra (``pip install "neurolang[llm]"``).
"""

from .agent import AgentResult, AgentStep, IclAgent, ReactAgent, make_agent
from .config import (
    AssistantConfig,
    LLMConfig,
    LLMConfigError,
    ProviderConfig,
    load_config,
)
from .providers import LLMError, make_client

__all__ = [
    "AgentResult", "AgentStep", "AssistantConfig", "IclAgent", "LLMConfig",
    "LLMConfigError", "LLMError", "ProviderConfig", "ReactAgent",
    "load_config", "make_agent", "make_client",
]
