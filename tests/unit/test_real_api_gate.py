"""Unit tests for app.observability.real_api_gate.require_real_api_authorization
-- the fail-closed gate every real-provider ``_build_*`` function in this
project must call as its FIRST statement.

See tests/unit/test_provider_authorization_gate_integration.py for the
per-script proof that the gate is actually wired in before provider
client construction.
"""

from __future__ import annotations

import pytest

from app.observability.real_api_gate import RealApiCallsNotAuthorizedError, require_real_api_authorization


class TestBlockedValues:
    @pytest.mark.parametrize(
        "value",
        [
            None,  # not set at all
            "",
            "0",
            "false",
            "False",
            "FALSE",
            "true",  # loosely-truthy but not the one exact required value
            "True",
            "yes",
            "y",
            "1 ",  # trailing whitespace
            " 1",  # leading whitespace
            "10",
            "01",
            "authorized",
        ],
    )
    def test_blocks_everything_except_the_exact_authorized_value(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("ARL_ALLOW_REAL_API_CALLS", raising=False)
        else:
            monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", value)

        with pytest.raises(RealApiCallsNotAuthorizedError, match="ARL_ALLOW_REAL_API_CALLS"):
            require_real_api_authorization("test operation")

    def test_error_message_names_the_operation_but_never_a_secret(self, monkeypatch):
        monkeypatch.delenv("ARL_ALLOW_REAL_API_CALLS", raising=False)
        with pytest.raises(RealApiCallsNotAuthorizedError) as exc_info:
            require_real_api_authorization("OpenAI LLM agent construction (scripts/run_eval.py)")
        message = str(exc_info.value)
        assert "OpenAI LLM agent construction (scripts/run_eval.py)" in message
        assert "sk-" not in message  # never echoes anything credential-shaped


class TestAuthorizedValue:
    def test_exact_value_one_authorizes(self, monkeypatch):
        monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", "1")
        require_real_api_authorization("test operation")  # must not raise

    def test_credential_presence_alone_never_authorizes(self, monkeypatch):
        """A real-shaped credential in the environment, with the
        authorization flag absent, must still block -- credential
        presence and execution authorization are checked completely
        independently."""

        monkeypatch.delenv("ARL_ALLOW_REAL_API_CALLS", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-a-real-key-for-this-test")
        monkeypatch.setenv("COHERE_API_KEY", "fake-cohere-key-for-this-test")

        with pytest.raises(RealApiCallsNotAuthorizedError):
            require_real_api_authorization("test operation")
