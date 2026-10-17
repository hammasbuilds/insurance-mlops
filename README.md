# insurance-mlops

The checks that decide whether a deployed insurance model was trained on valid data and is allowed to ship. We
built it for ML engineers and model-risk teams at insurers: a point-in-time feature store that cannot leak future
values into training, a training/serving skew detector, model cards that must be complete, a consent ledger that
enforces purpose limitation at read time, and a release gate that refuses rather than warns. Pure Python, zero
dependencies.

## What it does

- **Point-in-time feature store.** Every value carries when it became true and when it became knowable (reporting
  lag). `get_as_of(entity, feature, t)` returns what was known at `t`, so a claim count updated after the decision
  never reaches the training set. The store refuses a value recorded as knowable before it happened, and reports
  staleness.
- **Training/serving skew detection.** Compares the two pipelines directly and names missing or extra features,
  type changes, unit mismatches (for example years vs months, by name and by a 12x ratio), missingness gaps and
  values outside the training range. `safe_to_serve` is a blocking verdict.
- **Model cards** with required fields (`intended_use`, `out_of_scope`, `training_data`, `owner`, `limitations`).
  An incomplete card blocks release; a missing owner renders as `UNASSIGNED`.
- **Consent and purpose limitation at read time.** `ledger.permitted(cohort, "marketing")` returns only subjects
  who consented to that purpose. No record means no consent; withdrawal is not retroactive for a past query;
  purged values stay purged.
- **A release gate that refuses.** Reads the card, metrics (Gini, Brier), fairness (disparate impact) and skew
  verdict, lists every failure, treats a missing or NaN metric as a failure, and names itself in its output.
- **A CLI for CI and on-call**: `insurance-mlops skew` and `insurance-mlops gate`, exit 0 safe / 1 refused / 2 bad
  input, reading JSON Lines, JSON arrays or CSV.

Serving and rollback live in [model-serving-platform](https://github.com/hammasbuilds/model-serving-platform), drift
monitoring in [llm-observability-platform](https://github.com/hammasbuilds/llm-observability-platform) and
scorecards in [credit-risk-engine](https://github.com/hammasbuilds/credit-risk-engine).

## Sample input / output

`python demo.py` sends five release candidates of one claims model through the gate:

```console
$ python demo.py
INPUT
   v4.2.0  documented, reviewed, calibrated
   v4.3.0  no owner recorded
   v4.4.0  units changed between train and serve
   v4.5.0  barely separates, and unevenly
   v4.6.0  eval job crashed and wrote NaN

OUTPUT
   v4.2.0  documented, reviewed, calibrated        APPROVED
   v4.3.0  no owner recorded                       REFUSED
      - model card incomplete: owner
   v4.4.0  units changed between train and serve   REFUSED
      - training/serving skew in vehicle_age (unit): serving divided by 12 matches training - looks like years vs months
      - training/serving skew in vehicle_age (range): 91.7% of serving values fall outside the training range 1 to 12
   v4.5.0  barely separates, and unevenly          REFUSED
      - gini 0.080 below minimum 0.3
      - brier 0.310 above maximum 0.25
      - disparate impact 0.610 below 0.80
   v4.6.0  eval job crashed and wrote NaN          REFUSED
      - gini is NaN - the evaluation did not produce a value
      - brier is NaN - the evaluation did not produce a value
      - worst_disparate_impact not supplied

   The gate refuses. It does not warn and proceed, because a warning
   on a release pipeline is a release.
```

v4.4.0 has the same model and the same metrics as the approved v4.2.0. What failed is that `vehicle_age` arrives in
months at serving time and was trained in years, a fault no evaluation metric can see.

## Results

| | |
|---|---|
| Release candidates in the demo | 5: 1 approved, 4 refused, each refusal naming its condition |
| Skew caught on v4.4.0 | unit mismatch (12x) and 91.7% of serving values outside the training range 1 to 12 |
| Gate thresholds (defaults) | Gini >= 0.3, Brier <= 0.25, disparate impact >= 0.80, card complete and reviewed, skew safe |
| Tests | 118, under two seconds; leakage is asserted exactly, not sampled for |

## Tech stack

- Python 3.10+, standard library only; in-memory store
- `argparse` CLI (`insurance-mlops`, or `python -m insurance`), JSON / JSON Lines / CSV input
- pytest and ruff for development, hatchling for the build; `uv sync` works too

## Run it

```bash
git clone https://github.com/hammasbuilds/insurance-mlops
cd insurance-mlops
pip install -e ".[dev]"  # zero runtime dependencies; pytest and ruff for development
python demo.py
pytest -q                # 118 tests
```

Every call below runs as shown; paste the block into a `python` shell:

```python
from insurance import ConsentLedger, FeatureStore, ModelCard, ReleaseGate, detect_skew

# 1. Point-in-time correctness: a value written later must not leak into an earlier read.
store = FeatureStore()
store.write("cust-1", "prior_claims", 1, event_time=1)
store.write("cust-1", "prior_claims", 4, event_time=60)
store.get_as_of("cust-1", "prior_claims", 30).value          # 1, not 4

rows, labels = store.build_training_set([("cust-1", 30, 0)], ["prior_claims"])
rows[0]["prior_claims"]                                        # 1 - what was known at t=30

# 2. Skew: same feature, disagreeing units between the two pipelines.
training_rows = [{"vehicle_age_years": 5}] * 20
serving_rows = [{"vehicle_age_years": 60}] * 20               # looks like months
detect_skew(training_rows, serving_rows)["safe_to_serve"]     # False

# 3. A model card with a gap in it, and the release gate that reads it.
card = ModelCard(
    name="claims-triage", version="1.0", intended_use="triage new claims",
    out_of_scope=["pricing"], training_data="2020-2024 claims", owner="",
    limitations=["not validated outside urban regions"], reviewed_by="a.reviewer",
)
ReleaseGate().evaluate(
    card=card,
    metrics={"gini": 0.4, "brier": 0.2},
    fairness={"worst_disparate_impact": 0.85},
    skew={"safe_to_serve": True},
)["approved"]                                                   # False - card.owner is blank
```

On files, for CI and on-call:

```bash
insurance-mlops skew training.jsonl serving.jsonl [--json]   # or: python -m insurance ...
insurance-mlops gate --card card.json --metrics metrics.json \
    --fairness fairness.json --skew skew.json [--json]
```

A CSV cell and the same value in a JSONL export compare equal, including blank/NA/NaN/null as missing.

## Licence

MIT
