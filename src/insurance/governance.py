"""Governance: model cards, consent, and the release gate.

Three obligations that are usually documentation and are treated here as code, because
documentation drifts from reality and code does not.

**Model cards** describe what a model is for and - more usefully - what it is *not*
for. The out-of-scope section is the one that prevents a claims-triage model being
quietly repurposed for pricing.

**Consent and purpose limitation.** Pakistan's PDPA, like GDPR, ties personal data to
the purpose it was collected for. A dataset gathered for claims processing is not
available for marketing because it happens to be in the same warehouse. The ledger is
an append-only history, and `FeatureStore.reader` / `build_training_set(consent=...)`
check it inside the read path rather than leaving it to a policy document.

**The release gate** refuses rather than warns, and fails closed: a metric that is
missing, NaN or not a number is a failure, not a pass. A gate that emits a warning is
a gate that gets merged past on a Friday.
"""

from __future__ import annotations

import bisect
import math
import numbers
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime


class GovernanceError(RuntimeError):
    pass


def _blank(value: object) -> bool:
    """Empty, whitespace-only, or a list whose every item is blank."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list | tuple | set | frozenset):
        return all(_blank(item) for item in value)
    return not value


@dataclass
class ModelCard:
    name: str
    version: str
    intended_use: str = ""
    out_of_scope: list[str] = field(default_factory=list)
    training_data: str = ""
    evaluation: dict[str, float] = field(default_factory=dict)
    fairness: dict[str, float] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    owner: str = ""
    reviewed_by: str = ""
    created_at: float = field(default_factory=time.time)

    # Fields without which the card is decoration.
    REQUIRED = ("intended_use", "out_of_scope", "training_data", "owner", "limitations")

    def missing_fields(self) -> list[str]:
        # "  " and [""] are as empty as "" and []: a card that passes on whitespace
        # has been filled in by a linter, not a person.
        return [f for f in self.REQUIRED if _blank(getattr(self, f))]

    def is_complete(self) -> bool:
        return not self.missing_fields()

    def is_reviewed(self) -> bool:
        return not _blank(self.reviewed_by)

    def to_markdown(self) -> str:
        def text(value: str, default: str) -> str:
            return value.strip() if not _blank(value) else default

        def items(values: list[str]) -> list[str]:
            kept = [f"- {v.strip()}" for v in values if not _blank(v)]
            return kept or ["- _not stated_"]

        lines = [
            f"# Model card: {self.name} v{self.version}",
            "",
            f"**Owner:** {text(self.owner, 'UNASSIGNED')}",
            f"**Reviewed by:** {text(self.reviewed_by, 'NOT REVIEWED')}",
            "",
            "## Intended use",
            text(self.intended_use, "_not stated_"),
            "",
            "## Out of scope",
        ]
        lines += items(self.out_of_scope)
        lines += ["", "## Training data", text(self.training_data, "_not stated_"), ""]

        if self.evaluation:
            lines += ["## Evaluation", ""]
            lines += ["| metric | value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in sorted(self.evaluation.items())]
            lines += [""]

        if self.fairness:
            lines += ["## Fairness", ""]
            lines += ["| measure | value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in sorted(self.fairness.items())]
            lines += [""]

        lines += ["## Limitations"]
        lines += items(self.limitations)
        return "\n".join(lines)


# ---------------------------------------------------------------------------- consent


def _clock_kind(value: object) -> str:
    if isinstance(value, datetime):
        return "datetime (timezone-aware)" if value.tzinfo is not None else "datetime (naive)"
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise GovernanceError(f"a consent timestamp must be finite, got {value!r}")
        return "number"
    raise GovernanceError(
        f"a consent timestamp must be a number or a datetime, got {type(value).__name__} {value!r}"
    )


@dataclass(frozen=True)
class ConsentEvent:
    """One line of the ledger. Events are only ever appended."""

    subject: str
    action: str  # "grant" | "withdraw"
    purposes: frozenset[str]
    at: float


@dataclass(frozen=True)
class ConsentInterval:
    """A span during which one subject consented to one purpose."""

    subject: str
    purpose: str
    granted_at: float
    withdrawn_at: float | None = None

    def covers(self, at) -> bool:
        return self.granted_at <= at and (self.withdrawn_at is None or at < self.withdrawn_at)


@dataclass
class ConsentLedger:
    """Append-only consent history, answerable for any past moment.

    Every grant and withdrawal is appended as an event and never rewritten, so the
    answer to "was this lawful on 1 February?" does not change when the customer
    withdraws in March or re-consents in June. Two rules keep that true:

    - Events for a subject must arrive in time order. A backdated event would change
      the answer to a question already asked, so it is refused.
    - Granting adds purposes (it never silently drops the ones already held), and
      withdrawing a purpose that is not currently granted is an error rather than a
      move of the old withdrawal date.
    """

    events: list[ConsentEvent] = field(default_factory=list)
    _intervals: dict[tuple[str, str], list[ConsentInterval]] = field(default_factory=dict)
    _starts: dict[tuple[str, str], list] = field(default_factory=dict)
    _last_at: dict[str, object] = field(default_factory=dict)
    _purposes_of: dict[str, set[str]] = field(default_factory=dict)
    _clock: str | None = None

    # ---- writes

    def _stamp(self, subject: object, at) -> object:
        if not isinstance(subject, str) or not subject.strip():
            raise GovernanceError(f"subject must be a non-empty string, got {subject!r}")
        at = time.time() if at is None else at
        kind = _clock_kind(at)
        if self._clock is None:
            self._clock = kind
        elif kind != self._clock:
            raise GovernanceError(
                f"timestamp {at!r} is a {kind} but this ledger uses {self._clock}; "
                "pass `at` explicitly with one kind of timestamp"
            )
        last = self._last_at.get(subject)
        if last is not None and at < last:
            raise GovernanceError(
                f"{subject!r}: event at {at!r} is earlier than the last recorded event at "
                f"{last!r}; the ledger is append-only and a backdated event would rewrite "
                "the answer for past dates"
            )
        return at

    @staticmethod
    def _purposes(purposes: object) -> frozenset[str]:
        if isinstance(purposes, str) or not isinstance(purposes, Iterable):
            raise GovernanceError(
                f"purposes must be a set of names such as {{'claims'}}, got {purposes!r}"
            )
        out = frozenset(purposes)
        if not out:
            raise GovernanceError("purposes must name at least one purpose")
        for p in out:
            if not isinstance(p, str) or not p.strip():
                raise GovernanceError(f"each purpose must be a non-empty string, got {p!r}")
        return out

    def _open(self, subject: str, purpose: str) -> ConsentInterval | None:
        spans = self._intervals.get((subject, purpose))
        return spans[-1] if spans and spans[-1].withdrawn_at is None else None

    def grant(self, subject: str, purposes: Iterable[str], at=None) -> ConsentEvent:
        purposes = self._purposes(purposes)
        at = self._stamp(subject, at)
        for purpose in sorted(purposes):
            if self._open(subject, purpose) is not None:
                continue  # already granted; the original grant date stands
            key = (subject, purpose)
            self._intervals.setdefault(key, []).append(ConsentInterval(subject, purpose, at))
            self._starts.setdefault(key, []).append(at)
            self._purposes_of.setdefault(subject, set()).add(purpose)
        return self._append(ConsentEvent(subject, "grant", purposes, at))

    def withdraw(
        self, subject: str, purposes: Iterable[str] | None = None, at=None
    ) -> ConsentEvent:
        """Withdraw some purposes, or every purpose currently granted (the default)."""
        active = self.active_purposes(subject)
        if purposes is None:
            if not active:
                known = any(e.subject == subject for e in self.events)
                raise GovernanceError(
                    f"{subject!r} has no active consent to withdraw"
                    if known
                    else f"no consent on record for {subject!r}"
                )
            purposes = active
        purposes = self._purposes(purposes)
        not_held = sorted(purposes - active)
        if not_held:
            raise GovernanceError(
                f"{subject!r} does not currently consent to {', '.join(not_held)}; "
                "nothing to withdraw"
            )
        at = self._stamp(subject, at)
        for purpose in purposes:
            spans = self._intervals[(subject, purpose)]
            spans[-1] = ConsentInterval(subject, purpose, spans[-1].granted_at, at)
        return self._append(ConsentEvent(subject, "withdraw", purposes, at))

    def _append(self, event: ConsentEvent) -> ConsentEvent:
        self.events.append(event)
        self._last_at[event.subject] = event.at
        return event

    # ---- reads

    def active_purposes(self, subject: str) -> frozenset[str]:
        """Purposes the subject consents to as of their latest event."""
        known = self._purposes_of.get(subject, ())
        return frozenset(p for p in known if self._open(subject, p) is not None)

    def permits(self, subject: str, purpose: str, *, at=None) -> bool:
        """Did `subject` consent to `purpose` at moment `at` (default: now)?"""
        at = time.time() if at is None else at
        key = (subject, purpose)
        starts = self._starts.get(key)
        if not starts:
            return False
        if self._clock is not None and _clock_kind(at) != self._clock:
            raise GovernanceError(
                f"query time {at!r} is a {_clock_kind(at)} but this ledger uses {self._clock}"
            )
        index = bisect.bisect_right(starts, at)
        return bool(index) and self._intervals[key][index - 1].covers(at)

    def permitted(self, subjects: Iterable[str], purpose: str, *, at=None) -> list[str]:
        """Filter a cohort to those who consented to this purpose at `at`.

        Absence of a record is absence of consent. Defaulting the other way is how a
        marketing model ends up trained on claims data.
        """
        if isinstance(subjects, str):
            raise GovernanceError("subjects must be a list of subject ids, not one string")
        at = time.time() if at is None else at
        return [s for s in subjects if self.permits(s, purpose, at=at)]

    def history(self, subject: str) -> list[ConsentEvent]:
        """Every event for one subject, oldest first - the audit answer."""
        return [e for e in self.events if e.subject == subject]

    def intervals(self, subject: str) -> list[ConsentInterval]:
        known = self._purposes_of.get(subject, ())
        spans = [i for p in known for i in self._intervals[(subject, p)]]
        return sorted(spans, key=lambda i: (i.granted_at, i.purpose))


# ------------------------------------------------------------------------ release gate


def _number_problem(name: str, value: object) -> str | None:
    """Why `value` cannot be trusted as a metric, or None if it can."""
    if value is None:
        return f"{name} not supplied"
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return f"{name} is not a number (got {type(value).__name__} {value!r})"
    if math.isnan(value):
        return f"{name} is NaN - the evaluation did not produce a value"
    if math.isinf(value):
        return f"{name} is infinite - the evaluation did not produce a usable value"
    return None


def _skew_failures(skew: object) -> list[str]:
    if skew is None:
        return ["no training/serving skew report supplied"]
    if not isinstance(skew, Mapping):
        return [f"skew report is not a dict (got {type(skew).__name__})"]
    verdict = skew.get("safe_to_serve")
    if verdict is not True and verdict is not False:
        return ["skew report has no safe_to_serve verdict"]
    findings = [f for f in skew.get("findings") or [] if isinstance(f, Mapping)]
    critical = [f for f in findings if f.get("severity") == "critical"]
    count = skew.get("critical", len(critical))
    if verdict and not count and not critical:
        return []
    if verdict:
        # A report that says safe while counting critical findings is inconsistent,
        # and an inconsistent report is not evidence of anything.
        return [f"skew report says safe_to_serve but lists {count or len(critical)} critical"]
    if not critical:
        return [f"{count} critical training/serving skew finding(s) (details not supplied)"]
    return [
        f"training/serving skew in {f.get('feature', '?')} ({f.get('kind', '?')}): "
        f"{f.get('detail', '')}"
        for f in critical
    ]


@dataclass
class ReleaseGate:
    """Conditions a model must satisfy before it may serve.

    Fails closed. Each evidence requirement can be switched off explicitly, but a
    missing, NaN or non-numeric value is never read as a pass.
    """

    min_gini: float = 0.30
    max_brier: float = 0.25
    max_disparate_impact_gap: float = 0.20  # worst ratio must be >= 0.80
    require_complete_card: bool = True
    require_review: bool = True
    require_calibration: bool = True
    require_fairness: bool = True
    require_no_skew: bool = True

    NAME = "insurance-mlops/release/v2"

    def evaluate(
        self,
        *,
        card: ModelCard,
        metrics: Mapping | None,
        fairness: Mapping | None = None,
        skew: Mapping | None = None,
    ) -> dict:
        if not isinstance(card, ModelCard):
            raise GovernanceError(f"card must be a ModelCard, got {type(card).__name__}")
        failures: list[str] = []

        if self.require_complete_card and not card.is_complete():
            failures.append(f"model card incomplete: {', '.join(card.missing_fields())}")
        if self.require_review and not card.is_reviewed():
            failures.append("model card has not been reviewed")

        if metrics is not None and not isinstance(metrics, Mapping):
            failures.append(f"metrics is not a dict (got {type(metrics).__name__})")
            metrics = {}
        metrics = metrics or {}

        gini = metrics.get("gini")
        if problem := _number_problem("gini", gini):
            failures.append(problem)
        elif not -1 <= gini <= 1:
            failures.append(f"gini {gini} is outside [-1, 1] - is it a percentage?")
        elif gini < self.min_gini:
            failures.append(f"gini {gini:.3f} below minimum {self.min_gini}")

        brier = metrics.get("brier")
        if brier is None and not self.require_calibration:
            pass
        elif problem := _number_problem("brier", brier):
            failures.append(problem)
        elif not 0 <= brier <= 1:
            failures.append(f"brier {brier} is outside [0, 1]")
        elif brier > self.max_brier:
            # Calibration is checked separately from ranking: a model can rank well
            # and still be wrong about the level, and an insurer prices from the level.
            failures.append(f"brier {brier:.3f} above maximum {self.max_brier}")

        if fairness is not None and not isinstance(fairness, Mapping):
            failures.append(f"fairness is not a dict (got {type(fairness).__name__})")
            fairness = {}
        worst = (fairness or {}).get("worst_disparate_impact")
        floor = 1 - self.max_disparate_impact_gap
        if worst is None and not self.require_fairness:
            pass
        elif problem := _number_problem("worst_disparate_impact", worst):
            failures.append(problem)
        elif worst < 0:
            failures.append(f"worst_disparate_impact {worst} is negative")
        elif worst < floor:
            failures.append(f"disparate impact {worst:.3f} below {floor:.2f}")

        if self.require_no_skew:
            failures.extend(_skew_failures(skew))

        return {
            "approved": not failures,
            "failures": failures,
            # Named so the decision is attributable in an audit, not anonymous.
            "gate": self.NAME,
        }
