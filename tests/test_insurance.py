"""Insurance MLOps tests.

Point-in-time correctness is an exact property — a value either was knowable at a
moment or it was not — so leakage is asserted rather than sampled for.
"""

from __future__ import annotations

import pytest

from insurance.featurestore import FeatureStore, FeatureStoreError
from insurance.governance import (
    ConsentLedger,
    GovernanceError,
    ModelCard,
    ReleaseGate,
)
from insurance.skew import detect_skew


# Readable timestamps: t(1) is day 1.
def t(day: float) -> float:
    return day * 86_400.0


def complete_card(**kw) -> ModelCard:
    defaults = dict(
        name="claims-fraud",
        version="1.0",
        intended_use="Triage motor claims for manual review.",
        out_of_scope=["Pricing", "Underwriting declines"],
        training_data="Motor claims 2023-2025, 180k records.",
        limitations=["Not validated on commercial fleet policies."],
        owner="risk-analytics",
        reviewed_by="compliance",
    )
    return ModelCard(**{**defaults, **kw})


class TestPointInTime:
    def test_a_value_recorded_later_is_invisible(self):
        """The bug this module exists for: training on a number that did not exist
        when the decision had to be made."""
        store = FeatureStore()
        store.write("cust-1", "prior_claims", 1, event_time=t(1))
        store.write("cust-1", "prior_claims", 4, event_time=t(60))

        assert store.get_as_of("cust-1", "prior_claims", t(30)).value == 1

    def test_the_latest_knowable_value_wins(self):
        store = FeatureStore()
        for day, value in ((1, 1), (10, 2), (20, 3)):
            store.write("cust-1", "prior_claims", value, event_time=t(day))
        assert store.get_as_of("cust-1", "prior_claims", t(15)).value == 2

    def test_nothing_is_visible_before_the_first_write(self):
        store = FeatureStore()
        store.write("cust-1", "prior_claims", 1, event_time=t(10))
        assert store.get_as_of("cust-1", "prior_claims", t(5)) is None

    def test_reporting_lag_is_respected(self):
        """A claim filed Monday and entered Wednesday was not available to a model
        running on Tuesday. Pretending otherwise is the same bug wearing a hat."""
        store = FeatureStore()
        store.write("cust-1", "prior_claims", 5, event_time=t(10), available_at=t(13))
        assert store.get_as_of("cust-1", "prior_claims", t(11)) is None
        assert store.get_as_of("cust-1", "prior_claims", t(14)).value == 5

    def test_a_value_cannot_be_knowable_before_it_happens(self):
        store = FeatureStore()
        with pytest.raises(FeatureStoreError):
            store.write("cust-1", "x", 1, event_time=t(10), available_at=t(5))

    def test_out_of_order_writes_are_handled(self):
        """Backfills arrive out of order; the query must not care."""
        store = FeatureStore()
        store.write("cust-1", "x", 3, event_time=t(20))
        store.write("cust-1", "x", 1, event_time=t(1))
        store.write("cust-1", "x", 2, event_time=t(10))
        assert store.get_as_of("cust-1", "x", t(15)).value == 2

    def test_training_set_is_built_at_decision_time_not_label_time(self):
        """Joining on the label time hands the model every fact that accumulated
        while the outcome was being determined."""
        store = FeatureStore()
        store.write("c1", "prior_claims", 0, event_time=t(1))
        store.write("c1", "prior_claims", 9, event_time=t(90))  # after the decision

        rows, targets = store.build_training_set([("c1", t(30), 1)], ["prior_claims"])
        assert rows[0]["prior_claims"] == 0
        assert targets == [1]

    def test_an_unknown_feature_is_none_not_an_error(self):
        """An incomplete row must produce a decision, not an exception."""
        store = FeatureStore()
        assert store.vector_as_of("nobody", ["x", "y"], t(1)) == {"x": None, "y": None}

    def test_staleness_is_reported(self):
        """A point-in-time-correct feature can still be useless: a risk score last
        refreshed eight months ago is legitimate and fictional."""
        store = FeatureStore()
        store.write("c1", "score", 700, event_time=t(1))
        assert store.staleness("c1", "score", t(31)) == t(30)

    def test_staleness_of_an_unknown_feature_is_none(self):
        assert FeatureStore().staleness("c1", "score", t(1)) is None

    def test_history_for_returns_writes_sorted_by_availability(self):
        store = FeatureStore()
        store.write("c1", "score", 1, event_time=t(1))
        store.write("c1", "score", 2, event_time=t(2), available_at=t(5))
        history = store.history_for("c1", "score")
        assert [r.value for r in history] == [1, 2]

    def test_history_for_unknown_pair_is_empty(self):
        assert FeatureStore().history_for("nobody", "score") == []


class TestRetention:
    def test_purge_removes_values_before_the_cutoff(self):
        """Retention is a legal obligation, not housekeeping."""
        store = FeatureStore()
        store.write("c1", "x", 1, event_time=t(1))
        store.write("c1", "x", 2, event_time=t(100))
        assert store.purge_before(t(50)) == 1
        assert store.get_as_of("c1", "x", t(200)).value == 2

    def test_purging_everything_removes_the_series(self):
        store = FeatureStore()
        store.write("c1", "x", 1, event_time=t(1))
        store.purge_before(t(50))
        assert store.entities() == set()

    def test_purge_does_not_resurrect_old_values(self):
        store = FeatureStore()
        store.write("c1", "x", 1, event_time=t(1))
        store.write("c1", "x", 2, event_time=t(100))
        store.purge_before(t(50))
        # The pre-cutoff value is gone, so an early query finds nothing rather than
        # silently falling back to a value that has been legally deleted.
        assert store.get_as_of("c1", "x", t(10)) is None


class TestSkew:
    def test_a_unit_mismatch_is_named(self):
        """Training in years, serving in months. The most common skew there is, and
        the easiest to fix once somebody says the word 'units'."""
        training = [{"age": 30.0 + i} for i in range(50)]
        serving = [{"age": (30.0 + i) * 12} for i in range(50)]
        result = detect_skew(training, serving)
        assert any(f["kind"] == "unit" for f in result["findings"])
        assert not result["safe_to_serve"]

    def test_a_missing_feature_is_critical(self):
        result = detect_skew([{"a": 1.0, "b": 2.0}], [{"a": 1.0}])
        assert any(f["kind"] == "missing_feature" for f in result["findings"])
        assert not result["safe_to_serve"]

    def test_an_extra_feature_is_critical(self):
        result = detect_skew([{"a": 1.0}], [{"a": 1.0, "b": 2.0}])
        assert any(f["kind"] == "extra_feature" for f in result["findings"])

    def test_a_missingness_gap_is_critical(self):
        """Usually a join that silently became an inner join on one side."""
        training = [{"x": 1.0} for _ in range(100)]
        serving = [{"x": None} for _ in range(50)] + [{"x": 1.0} for _ in range(50)]
        result = detect_skew(training, serving)
        assert any(f["kind"] == "missingness" for f in result["findings"])

    def test_a_type_change_is_critical(self):
        result = detect_skew([{"x": 1.0}] * 20, [{"x": "1.0"}] * 20)
        assert any(f["kind"] == "type" for f in result["findings"])

    def test_out_of_range_values_are_a_warning(self):
        training = [{"x": float(i)} for i in range(100)]
        serving = [{"x": float(i + 500)} for i in range(100)]
        result = detect_skew(training, serving)
        assert any(f["kind"] == "range" for f in result["findings"])

    def test_identical_pipelines_are_clean(self):
        rows = [{"a": float(i), "b": float(i * 2)} for i in range(100)]
        result = detect_skew(rows, rows)
        assert result["findings"] == []
        assert result["safe_to_serve"]

    def test_safe_to_serve_is_a_verdict_not_a_trend(self):
        """Skew is not something to watch climb. It means the pipelines disagree
        today."""
        result = detect_skew([{"a": 1.0}] * 20, [{"b": 1.0}] * 20)
        assert result["safe_to_serve"] is False


class TestModelCard:
    def test_a_complete_card_passes(self):
        assert complete_card().is_complete()

    def test_the_out_of_scope_section_is_required(self):
        """It is the section that stops a claims-triage model being quietly
        repurposed for pricing."""
        card = complete_card(out_of_scope=[])
        assert not card.is_complete()
        assert "out_of_scope" in card.missing_fields()

    def test_every_required_field_is_checked(self):
        for field_name in ModelCard.REQUIRED:
            card = complete_card(
                **{field_name: [] if field_name in {"out_of_scope", "limitations"} else ""}
            )
            assert field_name in card.missing_fields()

    def test_markdown_names_an_unassigned_owner_rather_than_omitting_it(self):
        card = complete_card(owner="")
        assert "UNASSIGNED" in card.to_markdown()

    def test_markdown_includes_metrics_when_present(self):
        card = complete_card(evaluation={"gini": 0.42})
        assert "gini" in card.to_markdown()


class TestConsent:
    def test_purpose_limitation_is_enforced(self):
        """A dataset gathered for claims is not available for marketing because it
        happens to be in the same warehouse."""
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=t(1))
        assert ledger.permitted(["s1"], "claims", at=t(10)) == ["s1"]
        assert ledger.permitted(["s1"], "marketing", at=t(10)) == []

    def test_no_record_means_no_consent(self):
        """Defaulting the other way is how a marketing model ends up trained on
        claims data."""
        assert ConsentLedger().permitted(["unknown"], "claims", at=t(10)) == []

    def test_withdrawal_takes_effect(self):
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=t(1))
        ledger.withdraw("s1", at=t(20))
        assert ledger.permitted(["s1"], "claims", at=t(30)) == []

    def test_withdrawal_is_not_retroactive_for_a_past_query(self):
        """A model trained lawfully in March was lawful in March. Retroactive
        invalidation is a different obligation from deletion."""
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=t(1))
        ledger.withdraw("s1", at=t(20))
        assert ledger.permitted(["s1"], "claims", at=t(10)) == ["s1"]

    def test_consent_before_it_was_granted_is_refused(self):
        ledger = ConsentLedger()
        ledger.grant("s1", {"claims"}, at=t(10))
        assert ledger.permitted(["s1"], "claims", at=t(5)) == []

    def test_withdrawing_without_a_record_raises(self):
        with pytest.raises(GovernanceError):
            ConsentLedger().withdraw("nobody")


class TestReleaseGate:
    def test_a_good_model_is_approved(self):
        result = ReleaseGate().evaluate(
            card=complete_card(),
            metrics={"gini": 0.45, "brier": 0.12},
            fairness={"worst_disparate_impact": 0.92},
            skew={"safe_to_serve": True, "critical": 0},
        )
        assert result["approved"] and result["failures"] == []

    def test_an_incomplete_card_blocks_release(self):
        result = ReleaseGate().evaluate(card=complete_card(owner=""), metrics={"gini": 0.45})
        assert not result["approved"]
        assert any("incomplete" in f for f in result["failures"])

    def test_an_unreviewed_card_blocks_release(self):
        result = ReleaseGate().evaluate(card=complete_card(reviewed_by=""), metrics={"gini": 0.45})
        assert any("not been reviewed" in f for f in result["failures"])

    def test_weak_discrimination_blocks_release(self):
        result = ReleaseGate().evaluate(card=complete_card(), metrics={"gini": 0.05})
        assert any("gini" in f for f in result["failures"])

    def test_poor_calibration_blocks_release_even_with_good_ranking(self):
        """An insurer prices from the level, not the ranking."""
        result = ReleaseGate().evaluate(card=complete_card(), metrics={"gini": 0.60, "brier": 0.40})
        assert any("brier" in f for f in result["failures"])

    def test_a_fairness_gap_blocks_release(self):
        result = ReleaseGate().evaluate(
            card=complete_card(),
            metrics={"gini": 0.45},
            fairness={"worst_disparate_impact": 0.55},
        )
        assert any("disparate impact" in f for f in result["failures"])

    def test_critical_skew_blocks_release(self):
        result = ReleaseGate().evaluate(
            card=complete_card(),
            metrics={"gini": 0.45},
            skew={"safe_to_serve": False, "critical": 2},
        )
        assert any("skew" in f for f in result["failures"])

    def test_a_missing_metric_blocks_rather_than_defaults(self):
        """Absent evidence is not evidence of adequacy."""
        result = ReleaseGate().evaluate(card=complete_card(), metrics={})
        assert not result["approved"]

    def test_every_failure_is_listed_not_just_the_first(self):
        """One fix per deploy attempt is how a release takes a fortnight."""
        result = ReleaseGate().evaluate(
            card=complete_card(owner="", reviewed_by=""), metrics={"gini": 0.01}
        )
        assert len(result["failures"]) >= 3

    def test_the_gate_is_named_for_the_audit_trail(self):
        result = ReleaseGate().evaluate(card=complete_card(), metrics={"gini": 0.45})
        assert result["gate"] == "insurance-mlops/release/v2"
