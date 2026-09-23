"""Session-wide safety net: no test in this suite may open a real network
socket, in-process, ever -- regardless of whether a developer has a real
`.env` with real credentials sitting in the repository.

This is deliberately NOT opt-in. It is autouse, applied to every single
test automatically, so protection does not depend on any test author
remembering to use FakeLLMProvider or to mock a client. If a test
accidentally reaches a real network call path (e.g. by constructing a real
OpenAIProvider with a live-looking key and calling it), the connection
attempt itself raises NetworkBlockedError immediately, before any bytes
reach the network -- not a timeout, not a hang, not a real (rejected or
otherwise) HTTP request.

Scope note: this protects code running IN this pytest process. It does NOT
reach into a child process started via subprocess.run (used by the CLI
integration tests) -- a subprocess has its own independent socket module,
unaffected by monkeypatching done here. Those tests are protected by a
different, complementary mechanism: hermetic configuration (see
tests/integration/test_cli.py), which ensures such a subprocess never
acquires real-looking provider credentials in the first place, so it never
reaches a network-call code path regardless of this fixture.

Any future test that deliberately wants to exercise the real network path
must not rely on this suite at all -- see the module docstring in
tests/integration/test_cli.py for how normal pytest can never accidentally
promote itself into a live API test.
"""

from __future__ import annotations

import socket

import pytest


class NetworkBlockedError(RuntimeError):
    """Raised instead of allowing a real network connection during the
    automated test suite. Seeing this means a test reached a real
    network-call code path -- it should be using FakeLLMProvider or an
    explicit mock instead."""


def _blocked(*args, **kwargs):
    raise NetworkBlockedError(
        "Network access is blocked during the automated test suite. A test "
        "reached a real socket-connect call path (e.g. a real HTTP request "
        "toward OpenAI or any other host). Use FakeLLMProvider or a mock "
        "instead of a real provider/client in unit and integration tests."
    )


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Applied to every test automatically. Patches the lowest common layer
    (the stdlib socket module) so the barrier holds regardless of which
    HTTP library a future code path might use (openai's SDK, httpx,
    requests, urllib, or raw sockets) -- all of them ultimately call through
    socket.socket.connect / socket.create_connection to open a real
    connection.

    Does not affect subprocess.run: spawning a child process uses OS-level
    fork/exec, not this process's socket module, so the CLI integration
    tests (which invoke `python scripts/run_eval.py` as a subprocess) are
    unaffected by this patch and continue to work normally.
    """

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
