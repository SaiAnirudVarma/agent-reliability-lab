"""Provider-call pacing: enforces a minimum interval between the START
times of consecutive provider calls -- never between a response and the
next call -- so response/network latency is never double-counted against
the pacing budget.

Added after `docs/incidents/2026-09-24-cohere-reranking-rate-limit.md`:
Cohere's trial API key enforces 10 requests/minute server-side, and the
original reranking runner had no pacing at all.
"""

from __future__ import annotations

from typing import Callable, Optional


class CallPacer:
    """Semantics:

        next_allowed_start = previous_call_start + interval_seconds

    ``wait_for_next_call()`` sleeps ``max(0, next_allowed_start - now())``
    -- nothing at all on the very first call (there is no
    ``previous_call_start`` yet) -- then records the moment the call
    actually starts (read fresh from ``monotonic`` after any sleep) as the
    new ``previous_call_start``.

    Because ``previous_call_start`` is recorded at call *start*, not call
    *end*, a provider call that itself takes longer than
    ``interval_seconds`` naturally results in zero additional sleep before
    the next call -- the elapsed response time already satisfies the
    interval on its own.

    ``monotonic``/``sleep`` are injectable so tests never actually wait --
    see ``tests.support.fake_clock.FakeClock``.
    """

    def __init__(
        self,
        interval_seconds: float,
        monotonic: Callable[[], float],
        sleep: Callable[[float], None],
    ):
        if interval_seconds < 0:
            raise ValueError(f"interval_seconds must be >= 0, got {interval_seconds}")
        self._interval = interval_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._previous_call_start: Optional[float] = None

    def wait_for_next_call(self) -> None:
        """Call this immediately before each provider-level call -- never
        after. Never called at all after the final case's call."""

        now = self._monotonic()
        if self._previous_call_start is not None:
            next_allowed_start = self._previous_call_start + self._interval
            wait_seconds = next_allowed_start - now
            if wait_seconds > 0:
                self._sleep(wait_seconds)
                now = self._monotonic()
        self._previous_call_start = now
