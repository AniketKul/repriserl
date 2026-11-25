"""Inclusive elapsed time and disjoint phase durations from one monotonic clock."""
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import time


class PhaseMeter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._start = clock()
        self._active = False
        self._durations: dict[str, float] = {}

    def elapsed(self) -> float:
        return self._clock() - self._start

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not name or self._active:
            raise ValueError("phases must be named and cannot overlap")
        start = self._clock()
        self._active = True
        try:
            yield
        finally:
            self._durations[name] = self._durations.get(name, 0.0) + self._clock() - start
            self._active = False

    def finish(self) -> dict:
        if self._active:
            raise RuntimeError("cannot finish while a phase is running")
        return {"wall_seconds": self.elapsed(), "phase_seconds": self._durations.copy()}
