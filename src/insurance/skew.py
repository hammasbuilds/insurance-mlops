"""Training/serving skew detection.

Distinct from drift, and the distinction matters because the fixes are different.

**Drift** is the world changing. The model was right and reality moved; you retrain.
**Skew** is the two code paths disagreeing. Training computed `age` in years and
serving computes it in months, or training saw a cleaned column and serving sees the
raw one. The model was never right in production and retraining will not help.

Skew is the more common failure and the harder one to see, because every individual
component passes its own tests. It only appears when the two pipelines are compared
directly - which is what this module does.

What counts as the *same* value matters as much as what counts as different:

- `2` and `2.0` are the same number. JSON serialisers drop the `.0`, and a pandas int
  column with one NaN becomes float, so int vs float is not skew.
- `None` and `NaN` are both missing. pandas fills a broken join with NaN, not None, so
  a detector that only counts `None` passes a serving batch that is entirely empty.
"""

from __future__ import annotations

import bisect
import math
import numbers
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

# Share of serving values allowed to be categories never seen in training.
UNSEEN_WARNING_SHARE = 0.05
# Above this, most of serving speaks a different vocabulary: an encoding change.
UNSEEN_CRITICAL_SHARE = 0.50
MISSINGNESS_GAP = 0.10
NON_FINITE_GAP = 0.01
RANGE_SHARE = 0.05
# Most of serving outside anything training ever saw: the model is not being used
# on the population it was fitted on, whatever the cause.
RANGE_CRITICAL_SHARE = 0.50
# Rounding allowance for the rescaled comparison (whole years vs exact months).
UNIT_KS_FLOOR = 0.10

_UNIT_FACTORS = (
    (12, "years vs months"),
    (1 / 12, "months vs years"),
    (1000, "units vs thousands"),
    (1 / 1000, "thousands vs units"),
    (100, "fraction vs percent"),
    (1 / 100, "percent vs fraction"),
)


@dataclass
class SkewFinding:
    feature: str
    kind: str
    detail: str
    severity: str  # "critical" | "warning"


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    # numbers.Real covers float, int and numpy scalars. NaN is the missing one.
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            return math.isnan(value)
        except (TypeError, ValueError, OverflowError):
            return False
    # pandas.NA and pandas.NaT, recognised by name so pandas stays optional.
    return type(value).__name__ in {"NAType", "NaTType"}


def _kind(value: object) -> str:
    """The logical type of a present value. int and float are both 'number'."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, numbers.Real):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _numbers(values: Sequence) -> tuple[list[float], int]:
    """Finite numeric values, and how many were +/-inf."""
    finite: list[float] = []
    infinite = 0
    for v in values:
        if _kind(v) != "number" or _is_missing(v):
            continue
        f = float(v)
        if math.isinf(f):
            infinite += 1
        else:
            finite.append(f)
    return finite, infinite


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _preview(items, limit: int = 4) -> str:
    items = list(items)
    shown = ", ".join(repr(i) for i in items[:limit])
    return shown + (f", ... (+{len(items) - limit} more)" if len(items) > limit else "")


def _ks(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between the two CDFs."""
    a, b = sorted(a), sorted(b)
    gap = 0.0
    for x in set(a) | set(b):
        gap = max(gap, abs(bisect.bisect_right(a, x) / len(a) - bisect.bisect_right(b, x) / len(b)))
    return gap


def _ks_critical(n: int, m: int) -> float:
    """KS critical value at alpha = 0.01: gaps below this are sampling noise."""
    return 1.63 * math.sqrt((n + m) / (n * m))


def _unit_change(train: list[float], serve: list[float]) -> tuple[float, str] | None:
    """The conversion factor that turns serving back into training, if one does.

    A single mean ratio is too brittle: two independent 300-row samples of a 0-15
    years feature gave a months/years mean ratio of 13.2, far outside any tight
    tolerance around 12. So compare whole distributions instead. A unit change is
    declared when serving differs from training by more than sampling noise (KS
    above its alpha=0.01 critical value) and dividing serving by a known factor
    makes the two indistinguishable again (KS within noise, with a small floor for
    rounding, e.g. whole years vs exact months) and at most half the raw gap.
    The tolerance therefore scales with the batch sizes rather than being a fixed 5%.
    """
    crit = _ks_critical(len(train), len(serve))
    raw = _ks(train, serve)
    # Capped at 0.5 so tiny batches (where the critical value exceeds 1) can still
    # flag disjoint distributions.
    if raw <= min(crit, 0.5):
        return None
    # An integer-valued training feature is usually a truncated count (age in whole
    # years), so serving is also compared after truncating to whole units.
    whole = all(v.is_integer() for v in train)
    candidates = []
    for f, u in _UNIT_FACTORS:
        rescaled = [v / f for v in serve]
        gap = _ks(train, rescaled)
        if whole:
            gap = min(gap, _ks(train, [float(math.floor(v)) for v in rescaled]))
        candidates.append((gap, f, u))
    gap, factor, label = min(candidates)
    if gap <= max(crit, UNIT_KS_FLOOR) and gap <= raw / 2:
        return factor, label
    return None


def _compare_categories(name: str, train: list, serve: list) -> list[SkewFinding]:
    train_set = set(train)
    unseen = Counter(v for v in serve if v not in train_set)
    share = _share(sum(unseen.values()), len(serve))
    if share <= UNSEEN_WARNING_SHARE:
        return []
    normalised = {str(v).strip().casefold() for v in train_set}
    hint = ""
    if all(str(v).strip().casefold() in normalised for v in unseen):
        hint = " - differs from training only by case or whitespace"
    detail = (
        f"{share:.1%} of serving values are categories unseen in training: "
        f"{_preview(v for v, _ in unseen.most_common())} "
        f"(training had {_preview(sorted(train_set, key=str))}){hint}"
    )
    severity = "critical" if share > UNSEEN_CRITICAL_SHARE else "warning"
    return [SkewFinding(name, "category", detail, severity)]


def compare_feature(
    name: str, training: Sequence, serving: Sequence, *, tolerance: float = 0.25
) -> list[SkewFinding]:
    findings: list[SkewFinding] = []

    train_missing = _share(sum(map(_is_missing, training)), len(training))
    serve_missing = _share(sum(map(_is_missing, serving)), len(serving))
    if abs(train_missing - serve_missing) > MISSINGNESS_GAP:
        # Usually a join that silently became an inner join on one side.
        findings.append(
            SkewFinding(
                name,
                "missingness",
                f"missing (None/NaN) {train_missing:.1%} in training vs "
                f"{serve_missing:.1%} in serving",
                "critical",
            )
        )

    train_present = [v for v in training if not _is_missing(v)]
    serve_present = [v for v in serving if not _is_missing(v)]

    train_kinds = {_kind(v) for v in train_present}
    serve_kinds = {_kind(v) for v in serve_present}
    if train_kinds and serve_kinds and train_kinds != serve_kinds:
        findings.append(
            SkewFinding(
                name,
                "type",
                f"training {sorted(train_kinds)} vs serving {sorted(serve_kinds)}",
                "critical",
            )
        )

    # Categorical features: a re-encoding (M/F -> male/female) keeps the type and
    # changes every value, which no numeric check can see.
    train_cats = [v for v in train_present if _kind(v) in {"string", "bool"}]
    serve_cats = [v for v in serve_present if _kind(v) in {"string", "bool"}]
    if train_cats and serve_cats:
        findings.extend(_compare_categories(name, train_cats, serve_cats))

    train_values, train_inf = _numbers(training)
    serve_values, serve_inf = _numbers(serving)

    train_inf_share = _share(train_inf, train_inf + len(train_values))
    serve_inf_share = _share(serve_inf, serve_inf + len(serve_values))
    if serve_inf_share - train_inf_share > NON_FINITE_GAP:
        # Usually a division by zero in the serving feature code.
        findings.append(
            SkewFinding(
                name,
                "non_finite",
                f"{serve_inf_share:.1%} of serving values are infinite vs "
                f"{train_inf_share:.1%} in training",
                "critical",
            )
        )

    if not train_values or not serve_values:
        return findings

    # A unit change rescales every value by a constant factor. Worth naming
    # separately because it is the single most common skew and the easiest to fix
    # once someone says the word "units".
    unit = _unit_change(train_values, serve_values)
    if unit:
        factor, label = unit
        findings.append(
            SkewFinding(
                name,
                "unit",
                f"serving divided by {factor:.4g} matches training - looks like {label}",
                "critical",
            )
        )

    train_mean = statistics.fmean(train_values)
    serve_mean = statistics.fmean(serve_values)
    if train_mean and abs(serve_mean - train_mean) / abs(train_mean) > tolerance:
        findings.append(
            SkewFinding(
                name,
                "distribution",
                f"mean {train_mean:.4g} in training vs {serve_mean:.4g} in serving",
                "warning",
            )
        )

    low, high = min(train_values), max(train_values)
    out_of_range = sum(1 for v in serve_values if not low <= v <= high)
    if out_of_range / len(serve_values) > RANGE_SHARE:
        # The model is extrapolating, which for tree models means predicting from a
        # leaf that was fitted on nothing like this.
        findings.append(
            SkewFinding(
                name,
                "range",
                f"{out_of_range / len(serve_values):.1%} of serving values fall outside the "
                f"training range {low:.4g} to {high:.4g}",
                "critical"
                if out_of_range / len(serve_values) > RANGE_CRITICAL_SHARE
                else "warning",
            )
        )

    return findings


def _check_rows(rows: object, side: str) -> list[Mapping]:
    if isinstance(rows, Mapping | str | bytes) or not isinstance(rows, Sequence):
        raise TypeError(f"{side} must be a list of dicts (one per row), got {type(rows).__name__}")
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"{side} row {i} is {type(row).__name__}, expected a dict")
    return list(rows)


def detect_skew(training: list[dict], serving: list[dict], *, tolerance: float = 0.25) -> dict:
    """Compare two batches of feature vectors, feature by feature.

    Returns a JSON-serialisable report whose `safe_to_serve` is False whenever any
    finding is critical. An empty batch on either side is itself critical: a
    comparison of nothing is not evidence that the pipelines agree.
    """
    training = _check_rows(training, "training")
    serving = _check_rows(serving, "serving")

    findings: list[SkewFinding] = []
    for side, rows in (("training", training), ("serving", serving)):
        if not rows:
            findings.append(
                SkewFinding("*", "empty_batch", f"the {side} batch has no rows", "critical")
            )

    train_features = {k for row in training for k in row}
    serve_features = {k for row in serving for k in row}

    if training and serving:
        for name in sorted(train_features - serve_features, key=str):
            findings.append(
                SkewFinding(
                    name, "missing_feature", "present in training, absent in serving", "critical"
                )
            )
        for name in sorted(serve_features - train_features, key=str):
            findings.append(
                SkewFinding(
                    name, "extra_feature", "present in serving, absent in training", "critical"
                )
            )

    for name in sorted(train_features & serve_features, key=str):
        findings.extend(
            compare_feature(
                name,
                [row.get(name) for row in training],
                [row.get(name) for row in serving],
                tolerance=tolerance,
            )
        )

    critical = [f for f in findings if f.severity == "critical"]
    return {
        "findings": [asdict(f) for f in findings],
        "critical": len(critical),
        "warnings": len(findings) - len(critical),
        "rows": {"training": len(training), "serving": len(serving)},
        # A blocking verdict, not a dashboard tile. Skew is not something to watch
        # trend upward - it means the two pipelines disagree today.
        "safe_to_serve": not critical,
    }
