"""Five candidate models at the release gate. One ships.

    python demo.py

Each candidate fails a different way a deployed model actually breaks:
an undocumented card, training/serving skew, weak discrimination with unfair
outcomes, and an evaluation job that crashed and wrote NaN. No model training,
no network, no dependencies.
"""

import sys
from pathlib import Path

# Run from any directory, installed or not.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from insurance.governance import ModelCard, ReleaseGate  # noqa: E402
from insurance.skew import detect_skew  # noqa: E402

NAN = float("nan")


def card(**over) -> ModelCard:
    base = dict(
        name="motor-claims-severity",
        version="4.2.0",
        intended_use="Predict claim severity band at first notification of loss.",
        out_of_scope=["underwriting decisions", "fraud referral"],
        training_data="2019-2024 motor claims, UK book, 412k rows",
        limitations=["not validated on commercial fleet", "no telematics features"],
        owner="claims-analytics",
        reviewed_by="model-risk",
    )
    base.update(over)
    return ModelCard(**base)


# Training rows, and two serving batches for the skew check. Training stored
# claim_count as int; the serving JSON sends 2.0 - that is not skew and must not
# block. The broken batch sends vehicle_age in months where training used years.
TRAINING = [
    {"vehicle_age": float(1 + i % 12), "claim_count": i % 3, "fuel": ("petrol", "diesel")[i % 2]}
    for i in range(240)
]
SERVING_OK = [{**row, "claim_count": float(row["claim_count"])} for row in TRAINING]
SERVING_BROKEN = [{**row, "vehicle_age": row["vehicle_age"] * 12} for row in SERVING_OK]
skew_clean = detect_skew(TRAINING, SERVING_OK)
skew_broken = detect_skew(TRAINING, SERVING_BROKEN)

GOOD = dict(
    metrics={"gini": 0.45, "brier": 0.12},
    fairness={"worst_disparate_impact": 0.92},
)

CANDIDATES = [
    ("v4.2.0  documented, reviewed, calibrated", dict(card=card(), skew=skew_clean, **GOOD)),
    (
        "v4.3.0  no owner recorded",
        dict(card=card(version="4.3.0", owner=""), skew=skew_clean, **GOOD),
    ),
    (
        "v4.4.0  units changed between train and serve",
        dict(card=card(version="4.4.0"), skew=skew_broken, **GOOD),
    ),
    (
        "v4.5.0  barely separates, and unevenly",
        dict(
            card=card(version="4.5.0"),
            metrics={"gini": 0.08, "brier": 0.31},
            fairness={"worst_disparate_impact": 0.61},
            skew=skew_clean,
        ),
    ),
    (
        "v4.6.0  eval job crashed and wrote NaN",
        dict(
            card=card(version="4.6.0"),
            metrics={"gini": NAN, "brier": NAN},
            fairness={},
            skew=skew_clean,
        ),
    ),
]

WIDTH = max(len(label) for label, _ in CANDIDATES) + 2

print("INPUT")
for label, _ in CANDIDATES:
    print(f"   {label}")
print()

print("OUTPUT")
gate = ReleaseGate()
for label, kw in CANDIDATES:
    result = gate.evaluate(**kw)
    verdict = "APPROVED" if result["approved"] else "REFUSED"
    print(f"   {label:{WIDTH}} {verdict}")
    for f in result["failures"]:
        print(f"      - {f}")
print()
print("   The gate refuses. It does not warn and proceed, because a warning")
print("   on a release pipeline is a release.")
