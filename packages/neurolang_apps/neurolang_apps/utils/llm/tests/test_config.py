import pytest

from ..config import (
    CONFIG_ENV,
    EXAMPLE_CONFIG,
    LLMConfigError,
    config_path,
    load_config,
    parse_config,
)

PROVIDERS = {
    "providers": {
        "remote": {
            "api": "openai", "base_url": "https://llm.test/v1",
            "model": "m", "api_key_env": "TEST_LLM_KEY",
        },
        "local": {"base_url": "http://localhost:1/v1", "model": "small"},
    }
}


def test_example_configuration_is_valid():
    config = load_config(EXAMPLE_CONFIG)
    assert config.default_provider == "albert"
    assert config.provider().model == "gemma-4-31b-it"
    assert {
        "openrouter", "albert", "anthropic", "claude-code", "albert-deepseek",
        "openrouter-gemma", "openrouter-deepseek",
    } <= set(
        config.providers
    )
    assert config.providers["albert"].base_url == (
        "https://albert.api.etalab.gouv.fr/v1"
    )
    assert config.providers["claude-code"].api == "command"
    assert config.assistant.strategy == "react"


def test_lookup_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    assert config_path() == EXAMPLE_CONFIG
    user = tmp_path / ".config" / "neurolang" / "llm.yaml"
    user.parent.mkdir(parents=True)
    user.write_text("")
    assert config_path() == user
    monkeypatch.setenv(CONFIG_ENV, str(tmp_path / "env.yaml"))
    assert config_path() == tmp_path / "env.yaml"
    assert config_path(tmp_path / "x.yaml") == tmp_path / "x.yaml"


def test_default_provider_is_the_first():
    config = parse_config(PROVIDERS)
    assert config.provider().name == "remote"
    assert config.provider("local").api == "openai"
    assert config.provider("local").api_key() is None


def test_api_key_from_environment(monkeypatch):
    provider = parse_config(PROVIDERS).provider()
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="TEST_LLM_KEY"):
        provider.api_key()
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    assert provider.api_key() == "secret"


def test_inline_api_key_is_rejected():
    data = {"providers": {"p": {
        "base_url": "https://x/v1", "model": "m", "api_key": "secret",
    }}}
    with pytest.raises(LLMConfigError, match="api_key_env"):
        parse_config(data)


@pytest.mark.parametrize("entry, message", [
    ({"model": "m"}, "lacks base_url"),
    ({"base_url": "u", "model": "m", "api": "soap"}, "Invalid api"),
    ({"base_url": "u", "model": "m", "colour": 1}, "unknown keys: colour"),
    ({"api": "command"}, "lacks command"),
    ({"api": "command", "command": ["x"], "tool_calling": "native"},
     "only tool_calling: text"),
])
def test_invalid_providers(entry, message):
    with pytest.raises(LLMConfigError, match=message):
        parse_config({"providers": {"p": entry}})


def test_unknown_provider_lists_the_available_ones():
    with pytest.raises(LLMConfigError, match="Available providers: local"):
        parse_config(PROVIDERS).provider("nope")
    with pytest.raises(LLMConfigError, match="Unknown LLM provider"):
        parse_config({**PROVIDERS, "default_provider": "nope"})


def test_invalid_assistant_settings():
    with pytest.raises(LLMConfigError, match="Invalid strategy"):
        parse_config({**PROVIDERS, "assistant": {"strategy": "guess"}})
    with pytest.raises(LLMConfigError, match="Unknown assistant settings"):
        parse_config({**PROVIDERS, "assistant": {"steps": 3}})


def test_unreadable_and_invalid_files(tmp_path):
    with pytest.raises(LLMConfigError, match="Cannot read"):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("providers: [")
    with pytest.raises(LLMConfigError, match="Invalid YAML"):
        load_config(bad)
