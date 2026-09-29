"""Regression tests for the fail-open paths found in review.

Each test is a realistic input that used to produce a silent wrong answer: a gate
approving a crashed evaluation, a skew check passing an all-NaN serving batch, a
consent query about February changing after a June re-grant.
"""

from __future__ import annotations

import math

import pytest

from insurance.governance import ModelCard, ReleaseGate
from insurance.skew import detect_skew

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


class TestSkewEquivalences:
    def test_int_and_float_are_the_same_type(self):
        """JSON sends 2 for 2.0. That is not skew and must not block a release."""
        result = detect_skew([{"x": 1.0}, {"x": 2.5}] * 10, [{"x": 1}, {"x": 2}, {"x": 2.5}] * 10)
        assert result["safe_to_serve"]
        assert not any(f["kind"] == "type" for f in result["findings"])

    def test_numpy_scalars_count_as_numbers(self):
        np = pytest.importorskip("numpy")
        result = detect_skew(
            [{"x": float(i)} for i in range(20)], [{"x": np.int64(i)} for i in range(20)]
        )
        assert result["safe_to_serve"]

    def test_bool_is_not_a_number(self):
        result = detect_skew([{"x": 1}] * 20, [{"x": True}] * 20)
        assert any(f["kind"] == "type" for f in result["findings"])

    def test_nan_counts_as_missing(self):
        """pandas fills a broken join with NaN, not None."""
        training = [{"x": float(i)} for i in range(50)]
        result = detect_skew(training, [{"x": NAN}] * 50)
        assert not result["safe_to_serve"]
        assert any(f["kind"] == "missingness" for f in result["findings"])

    def test_nan_does_not_poison_the_unit_check(self):
        training = [{"age": float(20 + i % 40)} for i in range(100)]
        serving = [{"age": r["age"] * 12} for r in training[:90]] + [{"age": NAN}] * 10
        result = detect_skew(training, serving)
        assert any(f["kind"] == "unit" for f in result["findings"])

    def test_matching_nan_rates_are_clean(self):
        rows = [{"x": float(i) if i % 5 else NAN} for i in range(100)]
        assert detect_skew(rows, rows)["findings"] == []

    def test_infinity_in_serving_is_critical(self):
        training = [{"ratio": float(i)} for i in range(1, 101)]
        serving = [{"ratio": math.inf if i % 4 == 0 else float(i)} for i in range(1, 101)]
        result = detect_skew(training, serving)
        assert any(f["kind"] == "non_finite" for f in result["findings"])

    def test_zero_centred_unit_change_is_named(self):
        """A +/-1 score served as +/-12 has mean ~0 on both sides."""
        training = [{"s": (-1.0) ** i * (1 + i % 3)} for i in range(60)]
        serving = [{"s": r["s"] * 12} for r in training]
        result = detect_skew(training, serving)
        assert any(f["kind"] == "unit" for f in result["findings"])
        assert not result["safe_to_serve"]

    def test_empty_serving_batch_is_not_safe(self):
        assert not detect_skew([{"x": 1.0}], [])["safe_to_serve"]

    def test_rows_must_be_dicts(self):
        with pytest.raises(TypeError, match="row 0"):
            detect_skew([[1, 2]], [{"x": 1}])

    def test_the_report_is_json_serialisable(self):
        import json

        json.dumps(detect_skew([{"x": 1.0}] * 5, [{"x": NAN}] * 5), allow_nan=False)


class TestCategoricalSkew:
    def test_a_re_encoding_is_critical_and_named(self):
        """M/F became male/female. Same type, every value different."""
        result = detect_skew([{"g": "M"}, {"g": "F"}] * 20, [{"g": "male"}, {"g": "female"}] * 20)
        assert not result["safe_to_serve"]
        (finding,) = result["findings"]
        assert finding["kind"] == "category" and "'male'" in finding["detail"]

    def test_a_case_change_is_diagnosed(self):
        result = detect_skew([{"s": "Yes"}, {"s": "No"}] * 20, [{"s": "yes"}, {"s": "no"}] * 20)
        assert "case or whitespace" in result["findings"][0]["detail"]

    def test_a_few_new_categories_are_a_warning(self):
        training = [{"make": m} for m in ["ford", "vw", "kia"] * 30]
        serving = [{"make": m} for m in ["ford", "vw", "kia"] * 30] + [{"make": "byd"}] * 10
        result = detect_skew(training, serving)
        assert result["safe_to_serve"]
        assert [f["severity"] for f in result["findings"]] == ["warning"]

    def test_the_same_vocabulary_is_clean(self):
        rows = [{"fuel": f} for f in ["petrol", "diesel", "ev"] * 20]
        assert detect_skew(rows, list(reversed(rows)))["findings"] == []
