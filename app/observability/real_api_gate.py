"""Fail-closed gate for real external provider calls (OpenAI, Cohere, or
any future provider) from experiment CLIs.

This repository has had two separate incidents in which a real credential
being present in the environment -- specifically, one auto-loaded from
``.env`` -- was sufficient, on its own, to let a real provider call
happen when one was not intended (see
``docs/incidents/2026-09-23-llm-baseline-artifact-deletion.md`` and
``docs/incidents/2026-09-24-unauthorized-real-openai-calls-during-phase-8a-smoketest.md``).
Credential presence and execution authorization are deliberately kept as
two entirely separate concepts: a caller must pass BOTH an independent
"real calls are authorized" check (this module) AND whatever
provider-specific credential check already exists (``OPENAI_API_KEY``,
``COHERE_API_KEY``, ...) -- neither one substitutes for the other.

Every ``_build_*`` function in this project that constructs a real
provider client (``OpenAIProvider``, ``OpenAIEmbeddingProvider``,
``CohereReranker``, ...) must call ``require_real_api_authorization`` as
its FIRST statement, before constructing that client and before checking
for any credential -- not merely before *calling* the client. Provider
construction is normally a no-network operation throughout this project,
but the gate is placed before even that, so no future provider
implementation can accidentally make a network call from inside its own
``__init__`` and slip past this check.
"""

from __future__ import annotations

import os

_AUTHORIZATION_ENV_VAR = "ARL_ALLOW_REAL_API_CALLS"
_AUTHORIZED_VALUE = "1"


class RealApiCallsNotAuthorizedError(Exception):
    """Raised by ``require_real_api_authorization`` when real provider
    calls have not been explicitly authorized for this process. Always
    fails closed -- a caller must never construct or invoke a real
    provider client after this is raised, regardless of what credentials
    are present in the environment or in ``.env``."""


def require_real_api_authorization(operation: str) -> None:
    """Raises ``RealApiCallsNotAuthorizedError`` unless
    ``ARL_ALLOW_REAL_API_CALLS`` is set to EXACTLY ``"1"`` in the current
    process environment. Returns ``None`` (does nothing) when authorized.

    Deliberately narrow: missing, empty, ``"0"``, ``"false"``,
    ``"False"``, ``"true"`` (not the one exact required value), a
    surrounding-whitespace variant, or any other value all BLOCK. There
    is exactly one authorized value, so a typo or a loosely-truthy
    convention (e.g. ``"yes"``) can never accidentally authorize a real
    run.

    This check is completely independent of, and never satisfied by, any
    credential's mere presence (``OPENAI_API_KEY``, ``COHERE_API_KEY``,
    ...) -- a credential loaded from a real shell environment or from
    ``.env`` carries no authorization on its own. ``operation`` is a
    short, human-readable label (e.g. ``"OpenAI LLM agent construction"``)
    included in the error message only, for diagnosability -- it never
    varies this function's gating logic.
    """

    value = os.environ.get(_AUTHORIZATION_ENV_VAR)
    if value == _AUTHORIZED_VALUE:
        return
    raise RealApiCallsNotAuthorizedError(
        f"Refusing to proceed with a real provider call ({operation}): "
        f"{_AUTHORIZATION_ENV_VAR} must be set to exactly {_AUTHORIZED_VALUE!r} to authorize real "
        f"external API calls. Got {value!r}. A real credential being present (e.g. OPENAI_API_KEY, "
        "COHERE_API_KEY -- whether exported directly or loaded from .env) never authorizes a real "
        "call by itself; the two are checked completely independently. See "
        "docs/incidents/2026-09-24-unauthorized-real-openai-calls-during-phase-8a-smoketest.md."
    )
