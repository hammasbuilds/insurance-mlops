"""Regression tests for the fail-open paths found in review.

Each test is a realistic input that used to produce a silent wrong answer: a gate
approving a crashed evaluation, a skew check passing an all-NaN serving batch, a
consent query about February changing after a June re-grant.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from insurance.featurestore import FeatureStore, FeatureStoreError
from insurance.governance import ModelCard, ReleaseGate

NAN = float("nan")


def t(day: float) -> float:
    return day * 86_400.0


def card(**kw) -> ModelCard:
    base = dict(
        name="claims-fraud",
        version="1.0",
        intended_use="Triage motor claims for manual review.",
        out_of_scope=["Pricing"],
        training_data="Motor claims 2023-2025.",
        limitations=["Not validated on fleet policies."],
        owner="risk-analytics",
        reviewed_by="compliance",
    )
    return ModelCard(**{**base, **kw})


GOOD = dict(
    metrics={"gini": 0.45, "brier": 0.12},
    fairness={"worst_disparate_impact": 0.92},
    skew={"safe_to_serve": True, "critical": 0, "findings": []},
)


def gate(**over) -> dict:
    return ReleaseGate().evaluate(card=over.pop("card", card()), **{**GOOD, **over})


class TestFeatureStoreInputs:
    @pytest.mark.parametrize("event_time", [NAN, "2026-01-01", None, True])
    def test_bad_event_times_are_refused(self, event_time):
        with pytest.raises(FeatureStoreError):
            FeatureStore().write("c1", "x", 1, event_time=event_time)

    def test_mixed_clocks_are_refused_at_write_time(self):
        store = FeatureStore()
        store.write("c1", "x", 1, event_time=datetime(2026, 1, 1))
        with pytest.raises(FeatureStoreError, match="one kind of timestamp"):
            store.write("c2", "x", 1, event_time=t(1))

    def test_naive_and_aware_datetimes_do_not_mix(self):
        store = FeatureStore()
        store.write("c1", "x", 1, event_time=datetime(2026, 1, 1))
        with pytest.raises(FeatureStoreError):
            store.get_as_of("c1", "x", datetime(2026, 2, 1, tzinfo=timezone.utc))

    def test_datetime_staleness_is_a_timedelta(self):
        store = FeatureStore()
        store.write("c1", "x", 1, event_time=datetime(2026, 1, 1))
        assert store.staleness("c1", "x", datetime(2026, 1, 31)) == timedelta(days=30)

    @pytest.mark.parametrize("entity", ["", "  ", None, 7])
    def test_blank_entities_are_refused(self, entity):
        with pytest.raises(FeatureStoreError):
            FeatureStore().write(entity, "x", 1, event_time=0.0)

    def test_malformed_label_triples_are_refused(self):
        with pytest.raises(FeatureStoreError, match="triple"):
            FeatureStore().build_training_set([("c1", 1.0)], ["x"])

    def test_equal_availability_keeps_write_order(self):
        """A correction written at the same instant replaces the original."""
        store = FeatureStore()
        store.write("c1", "x", 1, event_time=t(1))
        store.write("c1", "x", 2, event_time=t(1))
        assert store.get_as_of("c1", "x", t(1)).value == 2

    def test_purge_after_out_of_order_writes(self):
        store = FeatureStore()
        for day in (30, 10, 20, 5):
            store.write("c1", "x", day, event_time=t(day))
        assert store.purge_before(t(15)) == 2
        assert [r.value for r in store.history_for("c1", "x")] == [20, 30]
        assert store.get_as_of("c1", "x", t(25)).value == 20

    def test_long_series_are_not_quadratic(self):
        """20k writes and reads to one series took over a minute before insort."""
        store = FeatureStore()
        n = 20_000
        start = time.perf_counter()
        for i in range(n):
            store.write("c1", "x", i, event_time=float((i * 7919) % n))
        for i in range(n):
            store.get_as_of("c1", "x", float(i))
        assert time.perf_counter() - start < 10  # ~0.5 s here; the old code took ~60 s
        assert store.get_as_of("c1", "x", float(n)).available_at == float(n - 1)
