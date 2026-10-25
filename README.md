# MeterMinder

MeterMinder is an open-source AI assistant for investigating household
electricity consumption. Phase 1 provides a tested daily aggregation pipeline
and an isolated TabPFN regression smoke test.

## Phase 1 setup

Python 3.13 is supported by the pinned dependencies. From the repository root,
create and activate a virtual environment, then install:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The UCI Individual Household Electric Power Consumption dataset is not
downloaded automatically. Download `household_power_consumption.txt` from the
[UCI dataset page](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
and place it under `data/raw/`.

Run the preprocessing function from Python:

```python
from meterminder.preprocessing import preprocess_household_data

preprocess_household_data(
    "data/raw/household_power_consumption.txt",
    "data/processed/daily_consumption.csv",
)
```

The output retains all observed days and records `valid_readings`,
`missing_readings`, `coverage_ratio`, and `is_complete`. Missing active-power
values are excluded from the energy sum, not converted to zero. Since the
source active power is a minute-average in kW, daily energy is computed as
`sum(valid kW) / 60`, yielding kWh.

Run tests with:

```powershell
python -m pytest
```

The isolated CPU smoke test uses two predictions from a 12-row synthetic
regression fixture. The first model use downloads the TabPFN checkpoint and
does not train on the electricity dataset:

```powershell
$env:PYTHONPATH = "src"
$env:TABPFN_NO_BROWSER = "1"
python -c "from meterminder.tabpfn_smoke import run_tabpfn_cpu_smoke_test; print(run_tabpfn_cpu_smoke_test())"
```

## TabPFN licensing and hardware note

The project pins TabPFN 9.2.0. The package code is Apache-2.0, but the default
TabPFN-3.5 weights have a separate TabPFN-3.5 Non-Commercial License. That
license permits non-commercial testing, evaluation, research, and qualifying
data-science competitions, but prohibits production/commercial use and
prohibits hosting the model as a free public service without a separate
commercial license. Therefore a public interactive MeterMinder demo using
these weights requires an explicit licensing decision before Phase 2.
Attribution must include: “The TABPFN-3.5 Model is licensed by Prior Labs GmbH
under the TABPFN-3.5 Non-Commercial License. Copyright © Prior Labs GmbH 2026.”

TabPFN documents CPU inference as feasible for moderate datasets and
recommends a GPU for performance. This Phase 1 smoke test is deliberately
tiny; it does not establish that a full daily dataset will fit comfortably in
16 GB RAM.

## Phase 2 LEAD anomaly benchmark

The LEAD files are local-only under `data/raw/lead/` and are intentionally
ignored by Git. Run the chunked audit and leakage-safe baseline from the
repository root:

```powershell
python scripts/run_lead_benchmark.py
```

This writes local `audit.json`, `metrics.json`, and validation prediction files
under `data/processed/lead/`. The pipeline sorts by building and timestamp,
validates lag timestamps across gaps, uses only shifted rolling statistics, and
holds out entire buildings. The downloaded `train_features.csv` and
`test_features.csv` are not merged: their `gte_*` columns are target-encoding
style features with unverified fit scope and are treated as potential leakage.

The optional hosted request is deliberately explicit and bounded:

```powershell
python scripts/run_lead_benchmark.py --run-tabpfn
```

It selects `v3.5_default`, uses at most 20,000 training and 5,000 validation
rows from disjoint building partitions, and makes one fit/predict request.
Check the account quota before running it; do not run this option from pytest.

For the corrected Phase 2C evaluation, use the dedicated building-disjoint
threshold/final split. This makes one additional controlled request only when
the saved predictions do not cover the representative final buildings:

```powershell
python scripts/run_lead_benchmark.py --correct-evaluation --run-tabpfn
```

The corrected results are stored under the ignored
`data/processed/lead/` directory and documented in
`docs/phase2_lead_report.md`.

Evaluate fixed-budget ranked alerts from the saved final predictions without
calling TabPFN:

```powershell
python scripts/diagnose_ranked_alerts.py
```

This writes `phase2e_ranked_alerts.json` and reports Average Precision plus
Precision@K and Recall@K for K values 25, 50, and 100.

## Phase 3 Streamlit MVP

Install the pinned dependencies, including Streamlit:

```powershell
python -m pip install -r requirements.txt
```

Start the local demo from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m streamlit run app.py
```

The app loads only the small processed prediction and metrics artifacts. It
does not call TabPFN or retrain models. The Investigation Queue is ranked by
the saved TabPFN score and supports budgets of 25, 50, and 100, building and
date filters, and a detail view. If `data/raw/lead/train.csv` is present, the
app performs a chunked lookup for observed meter readings and a small
neighboring timestamp context; it does not load the other raw files.

Investigation Mode hides ground-truth labels. Evaluation Mode exposes labels
and the held-out metrics for validation. The detector is not production-ready:
an anomaly does not prove energy waste, and the current TabPFN top-K results
are limited. No energy savings claims are made.

The upload workflow calculates descriptive statistics locally and can
optionally use a separately fitted hosted TabPFN model. Set up that reusable
account-bound model once, after checking quota, with:

```powershell
$env:PYTHONPATH = "src"
python scripts/setup_tabpfn_upload_model.py
```

The setup uses `train.csv` only, selects 100 deterministic rows from each of
200 buildings (20,000 training rows), explicitly selects `v3.5_default`, and
writes the small model reference and feature schema under the ignored
`data/processed/lead/` directory. It makes one hosted fit and is never called
by pytest or by the Streamlit app. The reference contains no model weights or
training data and can be loaded only by the same Prior Labs account while the
server-side fitted model remains available.

The inference route reuses `meterminder.lead.engineer_lead_features` and its
24-column ordered schema. It scores only rows with valid current readings;
missing or invalid readings remain unscored, and lag warm-up values remain
missing rather than being filled with zero. Generating scores sends uploaded
readings and derived features to Prior Labs for hosted processing. Scores are
experimental investigation signals, not proof of energy waste, equipment
failure, or savings. If the local model reference is absent or prediction
fails, descriptive analysis remains available without fabricated scores.

Upload CSV requirements:

- `timestamp` and `meter_reading` are required.
- `building_id` is optional.
- Readings must be non-negative numeric values; timestamps must parse.
- The app reports missing readings, duplicate timestamps, non-hourly gaps,
  chronological coverage, and the units caveat before analysis.
- Bills and PDFs are not accepted.