"""scheduled_coef: dynamic steering strength (Progressive Intensity Scheduling)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mcv.config import ScheduleSpec, SteeringSpec
from mcv.steering import scheduled_coef


def _spec(schedule=None, coef=-2.0) -> SteeringSpec:
    return SteeringSpec(vector_path="v.pt", layer=20, coef=coef, schedule=schedule)


def test_no_schedule_is_constant():
    spec = _spec()
    assert [scheduled_coef(spec, s, 100) for s in (1, 50, 100)] == [-2.0] * 3


def test_holds_then_ramps_to_end_multiple():
    spec = _spec(ScheduleSpec(start_fraction=0.2, end_multiple=3.0))
    assert scheduled_coef(spec, 1, 100) == -2.0
    assert scheduled_coef(spec, 20, 100) == -2.0  # last held step
    assert scheduled_coef(spec, 60, 100) == pytest.approx(-2.0 * 2.0)  # halfway up the ramp
    assert scheduled_coef(spec, 100, 100) == pytest.approx(-2.0 * 3.0)  # ends at end_multiple


def test_ramp_is_monotonic_and_keeps_sign():
    spec = _spec(ScheduleSpec(start_fraction=0.0, end_multiple=2.0), coef=1.5)
    vals = [scheduled_coef(spec, s, 50) for s in range(1, 51)]
    assert vals == sorted(vals) and all(v > 0 for v in vals)


def test_degenerate_total_steps_does_not_divide_by_zero():
    spec = _spec(ScheduleSpec(start_fraction=0.5, end_multiple=2.0))
    # 1-step run: finite, and between the start and end strength (denominator clamped to 1)
    assert -4.0 <= scheduled_coef(spec, 1, 1) <= -2.0


@pytest.mark.parametrize("kw", [{"start_fraction": 1.0}, {"start_fraction": -0.1}, {"end_multiple": 1.0}, {"bogus": 1}])
def test_schedule_spec_rejects_bad_values(kw):
    with pytest.raises(ValidationError):
        ScheduleSpec(**kw)
