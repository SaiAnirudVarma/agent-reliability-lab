"""Test-only network barrier: patches ``socket.socket.connect``/
``connect_ex`` to raise, so a test can assert zero real network
connections occurred as an extra defense-in-depth layer beyond
call-counting fakes -- see
docs/incidents/2026-09-24-unauthorized-real-openai-calls-during-phase-8a-smoketest.md,
which is exactly the failure mode this exists to make impossible to miss
in a test.
"""

from __future__ import annotations

import socket
from contextlib import contextmanager


class NetworkCallBlockedError(AssertionError):
    """Raised when code under test attempts a real network connection
    while the network barrier (``block_network``) is active."""


@contextmanager
def block_network():
    """Within this context, any attempt to open a real network socket
    connection raises ``NetworkCallBlockedError`` instead of connecting.
    Restores the original ``socket.socket`` methods on exit, even if the
    body raises."""

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def _blocked_connect(self, *args, **kwargs):
        raise NetworkCallBlockedError(f"blocked real network connection attempt: {args!r}")

    def _blocked_connect_ex(self, *args, **kwargs):
        raise NetworkCallBlockedError(f"blocked real network connection attempt: {args!r}")

    socket.socket.connect = _blocked_connect
    socket.socket.connect_ex = _blocked_connect_ex
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
