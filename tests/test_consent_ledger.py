"""Regression tests for the fail-open paths found in review.

Each test is a realistic input that used to produce a silent wrong answer: a gate
approving a crashed evaluation, a skew check passing an all-NaN serving batch, a
consent query about February changing after a June re-grant.
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from insurance.featurestore import ConsentRefused, FeatureStore, FeatureStoreError
from insurance.governance import ConsentLedger, GovernanceError, ModelCard, ReleaseGate

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


class TestConsentLedgerIsAppendOnly:
    def test_a_re_grant_does_not_rewrite_the_past(self):
        """Withdrew in March, re-consented in June: February stays lawful."""
        ledger = ConsentLedger()
        ledger.grant("s1", {"marketing"}, at=t(0))
        ledger.withdraw("s1", at=t(100))
        ledger.grant("s1", {"marketing"}, at=t(200))
        assert ledger.permitted(["s1"], "marketing", at=t(50)) == ["s1"]
        assert ledger.permitted(["s1"], "marketing", at=t(150)) == []
        assert ledger.permitted(["s1"], "marketing", at=t(250)) == ["s1"]

    def test_a_second_withdrawal_cannot_move_the_first(self):
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=t(0))
        ledger.withdraw("s1", at=t(100))
        with pytest.raises(GovernanceError, match="no active consent"):
            ledger.withdraw("s1", at=t(500))
        assert ledger.permitted(["s1"], "claims", at=t(200)) == []

    def test_a_backdated_event_is_refused(self):
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=t(10))
        with pytest.raises(GovernanceError, match="append-only"):
            ledger.withdraw("s1", at=t(5))

    def test_granting_a_new_purpose_keeps_the_old_ones(self):
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=t(0))
        ledger.grant("s1", {"marketing"}, at=t(10))
        assert ledger.permits("s1", "claims", at=t(20))
        assert ledger.active_purposes("s1") == {"claims", "marketing"}

    def test_re_granting_an_active_purpose_keeps_the_original_date(self):
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=t(0))
        ledger.grant("s1", {"claims"}, at=t(50))
        assert ledger.permits("s1", "claims", at=t(10))

    def test_withdrawing_one_purpose_keeps_the_others(self):
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims", "marketing"}, at=t(0))
        ledger.withdraw("s1", {"marketing"}, at=t(10))
        assert ledger.permits("s1", "claims", at=t(20))
        assert not ledger.permits("s1", "marketing", at=t(20))

    def test_withdrawing_a_purpose_never_granted_is_an_error(self):
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=t(0))
        with pytest.raises(GovernanceError, match="marketing"):
            ledger.withdraw("s1", {"marketing"}, at=t(1))

    def test_the_history_is_the_audit_trail(self):
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=t(0))
        ledger.withdraw("s1", at=t(5))
        assert [(e.action, e.at) for e in ledger.history("s1")] == [
            ("grant", t(0)),
            ("withdraw", t(5)),
        ]
        (span,) = ledger.intervals("s1")
        assert (span.granted_at, span.withdrawn_at) == (t(0), t(5))

    def test_a_string_of_purposes_is_refused_not_split_into_letters(self):
        with pytest.raises(GovernanceError, match="set of names"):
            ConsentLedger().grant("s1", "claims", at=0)

    @pytest.mark.parametrize("purposes", [set(), {""}, {"  "}, {1}])
    def test_empty_or_blank_purposes_are_refused(self, purposes):
        with pytest.raises(GovernanceError):
            ConsentLedger().grant("s1", purposes, at=0)

    @pytest.mark.parametrize("at", [NAN, "2026-01-01", True])
    def test_bad_timestamps_are_refused(self, at):
        with pytest.raises(GovernanceError):
            ConsentLedger().grant("s1", {"claims"}, at=at)

    def test_mixing_clocks_is_refused(self):
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=datetime(2026, 1, 1))
        with pytest.raises(GovernanceError, match="one kind of timestamp"):
            ledger.grant("s2", {"claims"}, at=t(1))

    def test_datetimes_work(self):
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=datetime(2026, 1, 1))
        assert ledger.permits("s1", "claims", at=datetime(2026, 2, 1))

    def test_default_time_is_now(self):
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"})
        assert ledger.permitted(["s1"], "claims") == ["s1"]
        assert not ledger.permits("s1", "claims", at=time.time() - 3600)


class TestConsentAtReadTime:
    @staticmethod
    def _setup():
        store = FeatureStore()
        store.write("s1", "prior_claims", 2, event_time=t(1))
        store.write("s2", "prior_claims", 5, event_time=t(1))
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=t(0))
        ledger.grant("s2", {"claims"}, at=t(0))
        ledger.withdraw("s2", at=t(40))
        return store, ledger

    def test_a_training_set_refuses_rows_without_consent_at_decision_time(self):
        store, ledger = self._setup()
        with pytest.raises(ConsentRefused, match="s2"):
            store.build_training_set(
                [("s1", t(50), 1), ("s2", t(50), 0)],
                ["prior_claims"],
                consent=ledger,
                purpose="claims",
            )

    def test_consent_is_judged_at_the_decision_time_not_now(self):
        store, ledger = self._setup()
        rows, targets = store.build_training_set(
            [("s1", t(30), 1), ("s2", t(30), 0)], ["prior_claims"], consent=ledger, purpose="claims"
        )
        assert rows == [{"prior_claims": 2}, {"prior_claims": 5}] and targets == [1, 0]

    def test_drop_mode_reports_what_it_dropped(self):
        store, ledger = self._setup()
        reader = store.reader(ledger, "claims")
        rows, targets = reader.build_training_set(
            [("s1", t(50), 1), ("s2", t(50), 0)], ["prior_claims"], on_refused="drop"
        )
        assert targets == [1]
        assert reader.refused == [("s2", t(50))]

    def test_a_single_read_is_refused_for_another_purpose(self):
        store, ledger = self._setup()
        with pytest.raises(ConsentRefused):
            store.reader(ledger, "marketing").get_as_of("s1", "prior_claims", t(30))

    def test_a_purpose_is_required_with_a_ledger(self):
        store, ledger = self._setup()
        with pytest.raises(FeatureStoreError, match="purpose"):
            store.build_training_set([], ["x"], consent=ledger)

    def test_a_purpose_without_a_ledger_is_refused(self):
        with pytest.raises(FeatureStoreError, match="ledger"):
            FeatureStore().build_training_set([], ["x"], purpose="claims")
