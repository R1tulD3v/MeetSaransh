"""Startup configuration validation.

The point of these tests is that a misconfigured deployment fails at boot with a clear
message, rather than at 3am on the first request that happens to exercise the setting.
"""

from __future__ import annotations

import pytest

from app import config


def test_the_default_configuration_is_valid():
    assert isinstance(config.validate(), list)


def test_a_missing_api_key_is_a_warning_not_a_failure(without_api_key):
    """Sample mode and lexical retrieval still work without a key, so booting is right."""
    warnings = config.validate()
    assert any("GROQ_API_KEY" in w for w in warnings)


def test_a_present_api_key_produces_no_key_warning(with_api_key):
    assert not any("GROQ_API_KEY" in w for w in config.validate())


def test_overlap_larger_than_the_chunk_budget_is_rejected(monkeypatch):
    """This combination makes the chunker unable to advance -- a hang, not a bad result."""
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 200)
    monkeypatch.setattr(config, "CHUNK_TARGET_WORDS", 100)
    with pytest.raises(config.ConfigError, match="CHUNK_OVERLAP_WORDS"):
        config.validate()


def test_a_nonsensical_top_k_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "RAG_TOP_K", 0)
    with pytest.raises(config.ConfigError, match="RAG_TOP_K"):
        config.validate()


def test_an_implausible_upload_limit_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(config.ConfigError, match="MAX_UPLOAD_BYTES"):
        config.validate()


def test_wildcard_cors_is_refused_in_production(monkeypatch):
    monkeypatch.setattr(config, "IS_PRODUCTION", True)
    monkeypatch.setattr(config, "CORS_ORIGINS", ["*"])
    with pytest.raises(config.ConfigError, match="CORS_ORIGINS"):
        config.validate()


def test_wildcard_cors_is_merely_unwise_in_development(monkeypatch):
    monkeypatch.setattr(config, "IS_PRODUCTION", False)
    monkeypatch.setattr(config, "CORS_ORIGINS", ["*"])
    config.validate()  # must not raise


def test_disabled_rate_limiting_in_production_is_flagged(monkeypatch, with_api_key):
    monkeypatch.setattr(config, "IS_PRODUCTION", True)
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", False)
    assert any("Rate limiting" in w for w in config.validate())


# ------------------------------------------------------------------- env var coercion
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("nonsense", False),
    ],
)
def test_boolean_env_vars_accept_the_usual_spellings(monkeypatch, raw, expected):
    monkeypatch.setenv("MS_TEST_FLAG", raw)
    assert config._env_bool("MS_TEST_FLAG", default=False) is expected


@pytest.mark.parametrize("default", [True, False])
def test_an_empty_boolean_env_var_keeps_the_default(monkeypatch, default):
    monkeypatch.setenv("MS_TEST_FLAG", "")
    assert config._env_bool("MS_TEST_FLAG", default=default) is default


def test_malformed_numeric_env_vars_fall_back_to_the_default(monkeypatch):
    """A typo in an env var should not take the process down at import time."""
    monkeypatch.setenv("MS_TEST_NUM", "not-a-number")
    assert config._env_int("MS_TEST_NUM", 42) == 42
    assert config._env_float("MS_TEST_NUM", 1.5) == 1.5


def test_unset_env_vars_use_their_defaults():
    assert config._env_int("MS_DEFINITELY_UNSET", 7) == 7
    assert config._env_str("MS_DEFINITELY_UNSET", "fallback") == "fallback"


def test_env_values_are_whitespace_trimmed(monkeypatch):
    monkeypatch.setenv("MS_TEST_STR", "  padded  ")
    assert config._env_str("MS_TEST_STR") == "padded"
