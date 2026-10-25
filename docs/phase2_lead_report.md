# Phase 2 LEAD benchmark report

## Scope

This phase audits the five local LEAD CSV files and benchmarks a
leakage-safe, building-level anomaly classifier. Raw Kaggle files remain under
`data/raw/lead/`, ignored by Git. No anomaly is interpreted as proven energy
waste, and no savings claim is made.

## Dataset evidence

The audit was performed in 200,000-row chunks and saved locally as
`data/processed/lead/audit.json`.

| File | Size (bytes) | Rows | Columns | Buildings | Missing values | Duplicate key rows |
|---|---:|---:|---:|---:|---|---:|
| `train.csv` | 57,680,722 | 1,749,494 | 4 | 200 | 107,653 `meter_reading` | 0 |
| `test.csv` | 69,106,465 | 1,800,567 | 4 | 206 | 94,864 `meter_reading` | 0 |
| `train_features.csv` | 582,046,560 | 1,749,494 | 58 | 200 | 107,653 `meter_reading` | 0 |
| `test_features.csv` | 608,172,758 | 1,800,567 | 58 | 206 | 94,864 `meter_reading` | 0 |
| `sample_submission.csv` | 18,695,143 | 1,800,567 | 2 | n/a | none | 0 |

Both timestamped datasets span `2016-01-01 00:00:00` through
`2016-12-31 23:00:00`. `train.csv` contains 1,712,198 normal and 37,296
anomalous rows: 2.131% anomalous. All 200 buildings contain both labels; no
building is all-normal or all-anomalous. The base-to-feature key comparison
found zero mismatches for all 1,749,494 train rows and all 1,800,567 test rows,
so the feature files are one-to-one in file order and would not duplicate rows.

The feature files add weather, building metadata, calendar encodings, and
`gte_*` target-encoding-style columns. Their fit scope is not documented by
the files, so those columns could use labels or validation-period information.
They were not merged. `building_id` is also not used as a numeric model
feature.

## Features and split

The initial model uses only `train.csv`: non-negative `meter_reading`,
missingness, `log1p(meter_reading)`, hour/day/month cyclic encodings, validated
1/24/168-hour lags, differences and zero-safe ratios, and a shifted
past-24-reading median with deviations. Lags are set missing when timestamps
are not exactly the expected interval. Rolling values use `shift(1)` before
rolling, so the current and future readings cannot enter the context.

`GroupShuffleSplit(random_state=42, test_size=0.2)` selected 160 training
buildings and 40 unseen validation buildings. The split contains 1,399,245
training rows and 350,249 validation rows, both spanning the full date range.
The normal/anomaly proportions are 97.899%/2.101% in training and
97.745%/2.255% in validation.

## Results

The local baseline is a class-balanced
`sklearn.SGDClassifier(loss="log_loss")`. On the full held-out building set:

| Model | Rows | ROC-AUC | Precision | Recall | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 350,249 | 0.6279 | 0.0350 | 0.7038 | 0.0666 | 1.0000 |

The threshold maximizes validation F1 and is therefore not an unbiased
threshold-selection estimate. The confusion matrix is
`[[188957, 153395], [2339, 5558]]`. The extreme threshold and low precision
show why ROC-AUC is the primary competition metric and why class imbalance
must be considered when interpreting precision/recall.

The one hosted request used `tabpfn-client==0.6.1`,
`TabPFNClassifier(model_path="v3.5_default")`, 20,000 training rows and the
same first 5,000 validation examples as the baseline comparison. It returned
finite probabilities and completed successfully:

| Model | Rows | ROC-AUC | Precision | Recall | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (same 5,000) | 5,000 | 0.7066 | 0.0386 | 0.9826 | 0.0744 | 1.0000 |
| TabPFN `v3.5_default` | 5,000 | 0.6925 | 0.1284 | 0.2435 | 0.1682 | 0.04493 |

TabPFN did not outperform the baseline on ROC-AUC for the same sample, but
did produce higher precision and F1 at its validation-selected threshold.
These are controlled benchmark results, not a tuned or final competition
submission.

## API, artifacts, and tests

The account check reported 70,000 of 20,000,000 credits used before inference.
The single request reported 1.736 seconds fit elapsed time and 1.944 seconds
of prediction-stage server timings. No retry or parameter tuning was done.

Generated local artifacts are ignored under `data/processed/lead/`:
`audit.json`, `metrics.json`, `baseline_predictions.csv`, and
`tabpfn_predictions.csv`. Focused and complete tests passed:

```text
python -m pytest -q
5 passed, 1 skipped
```

The skipped test is the pre-existing local TabPFN smoke test when its local
license/authentication path is unavailable. The hosted benchmark was run only
from the script, never from pytest.

Reproduce the local audit/baseline with:

```powershell
python scripts/run_lead_benchmark.py
```

The hosted call is explicit and must not be repeated casually:

```powershell
python scripts/run_lead_benchmark.py --run-tabpfn
```

Outstanding limitation: the `gte_*` feature provenance is unresolved, so the
feature files remain excluded until their construction history proves
label- and validation-safe.

The public Kaggle rules URL was checked, but the rules body is client-rendered
and was not available in the unauthenticated page response. Accordingly, no
raw or derived Kaggle data is redistributed in this repository.

## Phase 2C corrected evaluation

The earlier exploratory comparison above must not be treated as the final
benchmark: its TabPFN predictions were the first 5,000 validation rows, all
from building 108, and each model's threshold was selected on the rows used
for reporting.

The corrected evaluation preserves the original 160/40 building holdout and
splits the 40 held-out buildings into disjoint groups:

- Threshold-selection buildings (20):
  `121, 136, 139, 293, 623, 683, 685, 721, 844, 892, 919, 942, 969, 996,
  1143, 1172, 1239, 1258, 1264, 1284`
- Final-evaluation buildings (20):
  `108, 190, 560, 666, 680, 687, 693, 722, 730, 801, 889, 970, 974, 981,
  992, 1225, 1249, 1251, 1272, 1300`

Exactly 125 labelled rows were sampled per building using deterministic
per-building seeds. The threshold group has 2,500 rows (2,457 normal and 43
anomalous); the final group has 2,500 rows (2,445 normal and 55 anomalous).
The complete row indices and building lists are recorded in
`data/processed/lead/metrics.json`. The corrected baseline and TabPFN files
contain the same final row keys and labels.

Each threshold was selected only from the threshold group. The final metrics
were then computed without using final labels to tune the threshold:

| Model | Final rows | ROC-AUC | Precision | Recall | F1 | Selected threshold |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 2,500 | 0.6480 | 0.0371 | 0.7091 | 0.0705 | 1.0000 |
| TabPFN `v3.5_default` | 2,500 | 0.7170 | 0.0000 | 0.0000 | 0.0000 | 0.376778 |

Confusion matrices, in `[ [TN, FP], [FN, TP] ]` order:

- Baseline: `[[1433, 1012], [16, 39]]`
- TabPFN: `[[2401, 44], [55, 0]]`

On this corrected final sample, TabPFN outperformed the baseline on ROC-AUC
(0.7170 versus 0.6480), but not on precision, recall, or F1 at the
thresholds selected on the separate threshold group. This replaces the
earlier exploratory comparison as the appropriate benchmark result; it does
not establish generalization beyond this sampled final group.

The saved TabPFN predictions could not be reused because they covered only
building 108. Before the one additional request, the client reported model
`v3.5_default` available and account usage of 80,000/20,000,000 credits. No
model parameters were tuned and no repeated request was made. Corrected
artifacts are `corrected_baseline_predictions.csv` and
`corrected_tabpfn_predictions.csv`; the machine-readable summary is the
`corrected_evaluation` object in `metrics.json`.

Focused tests now cover disjoint evaluation buildings, non-overlapping
threshold/final rows, and class presence. The complete suite result is:

```text
python -m pytest -q
7 passed, 1 skipped
```

## Phase 2D threshold-transfer diagnostic

The Phase 2C artifacts were inspected locally without changing
`metrics.json` and without another hosted request. The saved TabPFN final
artifact contains 2,500 rows from all 20 final buildings. The earlier
statement that it predicted zero positives means zero **true positives**:
there were 44 predicted positives, all false positives. Recomputing the
confusion matrix from the saved score and prediction columns gives exactly
`[[2401, 44], [55, 0]]`. The baseline similarly has 1,051 predicted
positives and 39 true positives, matching `[[1433, 1012], [16, 39]]`.
Both score columns are finite, and both are treated as positive-class-1
probabilities.

Score distributions from the saved artifacts:

| Model/group | Rows | Score range | Normal median | Anomaly median |
|---|---:|---:|---:|---:|
| Baseline threshold buildings (all saved rows) | 174,757 | 0.0–1.0 | ~1.0000 | 1.0000 |
| Baseline final buildings | 2,500 | 0.0–1.0 | ~1.0000 | 1.0000 |
| TabPFN final buildings | 2,500 | 0.000040–0.733000 | 0.007583 | 0.029155 |

The baseline threshold-building score means are 0.7996 for normal rows and
0.8764 for anomalies; final-building means are 0.7729 and 0.8826. Baseline
building medians vary substantially in the threshold group (means range
approximately 0.588–0.962), showing building-level score calibration
differences. TabPFN threshold-building scores were not saved by the Phase 2C
runner, so their historical class/building distribution cannot be recovered
from the artifacts; only the final-group TabPFN distribution is directly
observable.

`precision_recall_curve` returns one extra precision/recall endpoint without a
corresponding threshold. The original helper correctly searches the complete
precision/recall arrays and clamps only if the best index is that final
endpoint; this is the standard scikit-learn index convention, not an
off-by-one pairing bug. There was no rounding before thresholds were
serialized. A focused test now protects this endpoint handling. The original
Phase 2C metrics and saved predictions remain unchanged.

The threshold failure is therefore primarily a transfer/calibration issue:

1. The baseline scores are heavily saturated at
   1.0 and score distributions differ by building. TabPFN has a much lower
   score scale and overlapping normal/anomaly distributions on the final
   buildings. A threshold optimized on only 43 anomalies across 20 other
   buildings is not guaranteed to transfer.

The smallest justified next step is a fresh, pre-registered evaluation:
save threshold-group scores for both models,
select thresholds only on a larger multi-building calibration group, and
evaluate once on a separate building-disjoint final group. Report the fixed
thresholds and score distributions before inspecting final labels. Do not
reuse the Phase 2C final labels to tune or claim improved performance.

## Phase 2E ranked-alert evaluation

This analysis uses only the saved Phase 2C final prediction artifacts. The
baseline and TabPFN files contain exactly the same 2,500 `row_index`,
`building_id`, `timestamp`, and `anomaly` values, covering the same 20
buildings. No threshold, model parameter, or final label was used to choose
an alert budget.

The final sample contains 55 anomalies out of 2,500 rows, so a random ranking
has expected precision `55 / 2500 = 0.022` (2.2%) at every fixed budget.
Rows were sorted by descending model score, then ascending building ID,
timestamp, and row index for deterministic ties.

| Model | Average Precision | K | Anomalies found | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.03431 | 25 | 1 | 0.0400 | 0.0182 |
| Baseline | 0.03431 | 50 | 1 | 0.0200 | 0.0182 |
| Baseline | 0.03431 | 100 | 4 | 0.0400 | 0.0727 |
| TabPFN | 0.04204 | 25 | 0 | 0.0000 | 0.0000 |
| TabPFN | 0.04204 | 50 | 1 | 0.0200 | 0.0182 |
| TabPFN | 0.04204 | 100 | 2 | 0.0200 | 0.0364 |

TabPFN has higher AP than the baseline and both AP values exceed the
prevalence reference, indicating useful global ranking signal. However, at
the fixed alert budgets tested, TabPFN is not better than the random reference
at K=25, K=50, or K=100; the baseline exceeds the random precision at K=25
and K=100. Therefore ranked alerts appear more useful than the failed
fixed-threshold TabPFN interaction as a general diagnostic concept, but these
results do **not** support TabPFN as a strong top-25/top-100 MVP queue yet.
The smallest defensible MVP next step is a ranked review queue with explicit
budgeting and monitoring, not a claim that the current TabPFN ranking is
production-ready.

The machine-readable ranked-alert output is
`data/processed/lead/phase2e_ranked_alerts.json`. The original Phase 2C and
Phase 2D metrics remain unchanged.
