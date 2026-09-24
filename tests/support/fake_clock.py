"""Test-only monotonic clock + sleep double: never actually waits.

``sleep(seconds)`` ADVANCES the fake clock's internal time by exactly
``seconds``, simulating a perfect sleep -- so pacing logic (see
``app.reranking.pacing.CallPacer``) can be tested deterministically and
instantly, with no real delay.
"""

from __future__ import annotations


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now_value = start
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self.now_value

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now_value += seconds

    def advance(self, seconds: float) -> None:
        """Advances the clock WITHOUT recording a sleep call -- simulates
        time elapsing for a reason other than this pacer's own sleep (e.g.
        a provider call's real response latency)."""

        self.now_value += seconds
