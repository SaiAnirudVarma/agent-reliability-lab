"""Unit tests for app.reranking.pacing.CallPacer. Uses FakeClock throughout
-- no test here ever actually sleeps.
"""

from __future__ import annotations

import pytest

from app.reranking.pacing import CallPacer
from tests.support.fake_clock import FakeClock


class TestCallPacer:
    def test_first_call_never_sleeps(self):
        clock = FakeClock(start=100.0)
        pacer = CallPacer(7.0, monotonic=clock.monotonic, sleep=clock.sleep)
        pacer.wait_for_next_call()
        assert clock.sleep_calls == []
        assert clock.now_value == 100.0

    def test_second_call_starts_no_earlier_than_interval_later(self):
        clock = FakeClock(start=0.0)
        pacer = CallPacer(7.0, monotonic=clock.monotonic, sleep=clock.sleep)
        pacer.wait_for_next_call()  # call 1 at t=0, no sleep
        pacer.wait_for_next_call()  # call 2: must sleep exactly 7.0
        assert clock.sleep_calls == [7.0]
        assert clock.now_value == 7.0

    def test_provider_latency_counts_toward_the_interval(self):
        """If the provider call between two wait_for_next_call() calls
        takes some time, that time reduces the required sleep."""
        clock = FakeClock(start=0.0)
        pacer = CallPacer(7.0, monotonic=clock.monotonic, sleep=clock.sleep)
        pacer.wait_for_next_call()  # call 1 starts at t=0
        clock.advance(3.0)  # the provider call itself took 3 seconds (t=3 now)
        pacer.wait_for_next_call()  # only 4 more seconds needed to reach t=7
        assert clock.sleep_calls == [4.0]
        assert clock.now_value == 7.0

    def test_provider_call_longer_than_interval_causes_no_sleep(self):
        clock = FakeClock(start=0.0)
        pacer = CallPacer(7.0, monotonic=clock.monotonic, sleep=clock.sleep)
        pacer.wait_for_next_call()  # call 1 at t=0
        clock.advance(10.0)  # provider call took LONGER than the interval
        pacer.wait_for_next_call()  # already past next_allowed_start -- no sleep
        assert clock.sleep_calls == []
        assert clock.now_value == 10.0

    def test_provider_call_exactly_equal_to_interval_causes_no_sleep(self):
        clock = FakeClock(start=0.0)
        pacer = CallPacer(7.0, monotonic=clock.monotonic, sleep=clock.sleep)
        pacer.wait_for_next_call()
        clock.advance(7.0)
        pacer.wait_for_next_call()
        assert clock.sleep_calls == []

    def test_three_consecutive_calls_each_paced_correctly(self):
        clock = FakeClock(start=0.0)
        pacer = CallPacer(7.0, monotonic=clock.monotonic, sleep=clock.sleep)
        pacer.wait_for_next_call()  # t=0
        pacer.wait_for_next_call()  # sleeps to t=7
        pacer.wait_for_next_call()  # sleeps to t=14
        assert clock.sleep_calls == [7.0, 7.0]
        assert clock.now_value == 14.0

    def test_zero_interval_never_sleeps(self):
        clock = FakeClock(start=0.0)
        pacer = CallPacer(0.0, monotonic=clock.monotonic, sleep=clock.sleep)
        pacer.wait_for_next_call()
        pacer.wait_for_next_call()
        assert clock.sleep_calls == []

    def test_negative_interval_rejected(self):
        clock = FakeClock()
        with pytest.raises(ValueError, match="interval_seconds"):
            CallPacer(-1.0, monotonic=clock.monotonic, sleep=clock.sleep)
