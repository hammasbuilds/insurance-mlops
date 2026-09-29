"""Point-in-time correct feature store.

The most expensive bug in production machine learning, and the one that almost never
shows up in a notebook:

    claim filed         2026-03-01
    fraud confirmed     2026-05-14
    feature `prior_claims_count` recomputed nightly, current value 4

Training on today's `prior_claims_count` teaches the model a number that **did not
exist** when the decision had to be made. Offline accuracy is excellent. Online
accuracy collapses, and the gap is blamed on drift.

A point-in-time join fixes it by asking a different question. Not *"what is this
customer's claim count?"* but *"what was it, as far as anyone knew, at 09:14 on the
first of March?"*

Every feature value here carries the moment it **became knowable**, which is not the
moment it became true. A claim filed on Monday and entered into the system on
Wednesday was not available to a model running on Tuesday, and pretending otherwise is
the same bug wearing a hat.

Timestamps are either numbers (epoch seconds, or any consistent numeric clock) or
`datetime` objects. One store uses one kind; mixing them is refused at write time
rather than surfacing as a comparison error deep inside a query.
"""

from __future__ import annotations

import bisect
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


class FeatureStoreError(ValueError):
    pass


class ConsentRefused(FeatureStoreError):
    """A read was refused because the subject had not consented to this purpose."""


class ConsentSource(Protocol):
    def permits(self, subject: str, purpose: str, *, at) -> bool: ...


def _time_kind(value: object, what: str, error: type[Exception] = FeatureStoreError) -> str:
    """Validate a timestamp and name its kind, so mixed clocks are caught early."""
    if isinstance(value, datetime):
        return "datetime (timezone-aware)" if value.tzinfo is not None else "datetime (naive)"
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise error(f"{what} must be a finite number, got {value!r}")
        return "number"
    raise error(
        f"{what} must be a number (e.g. epoch seconds) or a datetime, "
        f"got {type(value).__name__} {value!r}"
    )


def _require_name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeatureStoreError(f"{what} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class FeatureValue:
    entity: str
    feature: str
    value: float | str | None
    # When the fact became true in the world.
    event_time: float
    # When the system could first have known it. Never earlier than event_time.
    available_at: float

    def __post_init__(self) -> None:
        kind = _time_kind(self.event_time, f"{self.feature}: event_time")
        if _time_kind(self.available_at, f"{self.feature}: available_at") != kind:
            raise FeatureStoreError(
                f"{self.feature}: event_time and available_at use different clocks "
                f"({type(self.event_time).__name__} vs {type(self.available_at).__name__})"
            )
        if self.available_at < self.event_time:
            raise FeatureStoreError(
                f"{self.feature}: available_at precedes event_time - a value cannot be "
                "knowable before it happens"
            )


@dataclass
class FeatureStore:
    """Append-only history per (entity, feature), queried as of a moment."""

    _history: dict[tuple[str, str], list[FeatureValue]] = field(default_factory=dict)
    # Parallel to _history: the availability times, kept sorted so a write is one
    # insort and a read is one bisect, instead of a full sort or rebuild each time.
    _times: dict[tuple[str, str], list] = field(default_factory=dict)
    _clock: str | None = None

    def _check_clock(self, value: object, what: str) -> None:
        kind = _time_kind(value, what)
        if self._clock is None:
            self._clock = kind
        elif kind != self._clock:
            raise FeatureStoreError(
                f"{what} is a {kind} but this store's timestamps are {self._clock}; "
                "use one kind of timestamp per store"
            )

    def write(
        self,
        entity: str,
        feature: str,
        value,
        event_time: float,
        available_at: float | None = None,
    ) -> FeatureValue:
        _require_name(entity, "entity")
        _require_name(feature, "feature")
        record = FeatureValue(
            entity=entity,
            feature=feature,
            value=value,
            event_time=event_time,
            available_at=event_time if available_at is None else available_at,
        )
        self._check_clock(record.available_at, f"{feature}: available_at")
        key = (entity, feature)
        times = self._times.setdefault(key, [])
        series = self._history.setdefault(key, [])
        # Sorted by availability, because availability is what a query filters on.
        # bisect_right keeps writes with equal availability in write order, so the
        # later write wins a tie.
        index = bisect.bisect_right(times, record.available_at)
        times.insert(index, record.available_at)
        series.insert(index, record)
        return record

    def get_as_of(self, entity: str, feature: str, as_of: float) -> FeatureValue | None:
        """The most recent value that was *knowable* at `as_of`.

        Binary search on availability. Values whose `available_at` is later are
        invisible, which is the entire point - a value the system could not have read
        cannot be a feature.
        """
        times = self._times.get((entity, feature))
        if self._clock is not None:
            self._check_clock(as_of, "as_of")
        else:
            _time_kind(as_of, "as_of")
        if not times:
            return None
        index = bisect.bisect_right(times, as_of)
        return self._history[(entity, feature)][index - 1] if index else None

    def vector_as_of(self, entity: str, features: Sequence[str], as_of: float) -> dict[str, object]:
        if isinstance(features, str):
            raise FeatureStoreError("features must be a list of names, not a single string")
        return {
            f: (v.value if (v := self.get_as_of(entity, f, as_of)) is not None else None)
            for f in features
        }

    def build_training_set(
        self,
        labels: Sequence[tuple[str, float, int]],
        features: Sequence[str],
        *,
        consent: ConsentSource | None = None,
        purpose: str | None = None,
        on_refused: str = "raise",
    ) -> tuple[list[dict], list[int]]:
        """Assemble a training set from (entity, decision_time, label) triples.

        `decision_time` is when the model would have had to act - **not** when the
        label became known. Joining on the label time is the leak: it hands the model
        every fact that accumulated while the outcome was being determined.

        With `consent` and `purpose`, every row is checked against the ledger *at its
        decision time* before any feature is read. A subject without consent for that
        purpose at that moment raises `ConsentRefused` (the default), or with
        `on_refused="drop"` is left out; use `PurposeBoundReader` directly to see
        which rows were dropped.
        """
        if consent is not None or purpose is not None:
            reader = self.reader(consent, purpose)
            return reader.build_training_set(labels, features, on_refused=on_refused)
        rows, targets = [], []
        for entity, decision_time, label in _triples(labels):
            rows.append(self.vector_as_of(entity, features, decision_time))
            targets.append(label)
        return rows, targets

    def reader(self, consent: ConsentSource | None, purpose: str | None) -> PurposeBoundReader:
        """A view of this store that refuses reads the ledger does not permit."""
        return PurposeBoundReader(self, consent, purpose)

    def staleness(self, entity: str, feature: str, as_of: float) -> float | None:
        """How old the value in use is: seconds for a numeric clock, a timedelta
        for datetimes.

        Reported because a point-in-time-correct feature can still be useless: a
        risk score last refreshed eight months ago is technically legitimate and
        practically fiction.
        """
        record = self.get_as_of(entity, feature, as_of)
        return None if record is None else as_of - record.available_at

    def purge_before(self, cutoff: float) -> int:
        """Delete values whose availability predates `cutoff`.

        Retention is a legal obligation, not housekeeping. Pakistan's PDPA requires
        personal data to be kept no longer than the purpose needs, so purging has to
        be a first-class operation rather than a script somebody remembers to run.
        """
        if self._clock is not None:
            self._check_clock(cutoff, "cutoff")
        removed = 0
        for key, times in list(self._times.items()):
            start = bisect.bisect_left(times, cutoff)
            removed += start
            if start == len(times):
                del self._times[key]
                del self._history[key]
            elif start:
                del times[:start]
                del self._history[key][:start]
        return removed

    def entities(self) -> set[str]:
        return {entity for entity, _ in self._history}

    def history_for(self, entity: str, feature: str) -> list[FeatureValue]:
        """The full write history for one (entity, feature), sorted by availability."""
        return list(self._history.get((entity, feature), []))


def _triples(labels) -> list[tuple]:
    out = []
    for i, item in enumerate(labels):
        if not isinstance(item, tuple | list) or len(item) != 3:
            raise FeatureStoreError(
                f"labels[{i}] must be an (entity, decision_time, label) triple, got {item!r}"
            )
        out.append(tuple(item))
    return out


@dataclass
class PurposeBoundReader:
    """Reads from a FeatureStore on behalf of one purpose, checked against consent.

    This is what "consent enforced at read time" means here: the check sits in the
    read path, at the moment the value is read for, so a caller cannot get a feature
    value for a subject who had not consented to this purpose at that moment.
    """

    store: FeatureStore
    consent: ConsentSource | None
    purpose: str | None
    refused: list[tuple[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.consent is None or not hasattr(self.consent, "permits"):
            raise FeatureStoreError("a consent ledger is required for a purpose-bound read")
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise FeatureStoreError("purpose must be a non-empty string, e.g. 'claims'")

    def _check(self, entity: str, at) -> None:
        if not self.consent.permits(entity, self.purpose, at=at):
            raise ConsentRefused(
                f"{entity!r} had not consented to {self.purpose!r} at {at!r}; the read was refused"
            )

    def get_as_of(self, entity: str, feature: str, as_of) -> FeatureValue | None:
        self._check(entity, as_of)
        return self.store.get_as_of(entity, feature, as_of)

    def vector_as_of(self, entity: str, features: Sequence[str], as_of) -> dict[str, object]:
        self._check(entity, as_of)
        return self.store.vector_as_of(entity, features, as_of)

    def build_training_set(
        self, labels, features: Sequence[str], *, on_refused: str = "raise"
    ) -> tuple[list[dict], list[int]]:
        """Like `FeatureStore.build_training_set`, with every row consent-checked at
        its decision time. Dropped rows are recorded in `self.refused` as
        (entity, decision_time)."""
        if on_refused not in {"raise", "drop"}:
            raise FeatureStoreError(f"on_refused must be 'raise' or 'drop', got {on_refused!r}")
        self.refused = []
        rows, targets = [], []
        for entity, decision_time, label in _triples(labels):
            if not self.consent.permits(entity, self.purpose, at=decision_time):
                if on_refused == "raise":
                    raise ConsentRefused(
                        f"{entity!r} had not consented to {self.purpose!r} at decision "
                        f"time {decision_time!r}; pass on_refused='drop' to leave such "
                        "rows out"
                    )
                self.refused.append((entity, decision_time))
                continue
            rows.append(self.store.vector_as_of(entity, features, decision_time))
            targets.append(label)
        return rows, targets
