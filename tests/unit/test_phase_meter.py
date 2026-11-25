import pytest
from repriserl.metrics import PhaseMeter


def test_totals_keep_waits_and_failed_work_without_overlapping():
    clock = iter([0, 1, 3, 5, 8, 10])
    meter = PhaseMeter(lambda: next(clock))
    with meter.phase("train"):
        with pytest.raises(ValueError):
            with meter.phase("score"):
                pass
    with pytest.raises(ArithmeticError), meter.phase("train"):
        raise ArithmeticError
    assert meter.finish() == {"wall_seconds": 10, "phase_seconds": {"train": 5}}
