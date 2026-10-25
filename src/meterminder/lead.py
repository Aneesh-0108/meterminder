"""Leakage-safe audit, features, split, and baseline utilities for LEAD."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from itertools import zip_longest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    average_precision_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

LEAD_COLUMNS = {
    "train.csv": ["building_id", "timestamp", "meter_reading", "anomaly"],
    "test.csv": ["row_id", "building_id", "timestamp", "meter_reading"],
    "sample_submission.csv": ["row_id", "anomaly"],
}
KEY_COLUMNS = ["building_id", "timestamp"]
TARGET_COLUMN = "anomaly"
DEFAULT_CHUNK_SIZE = 200_000


def _timestamp_range(values: pd.Series) -> tuple[str | None, str | None]:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().sum() == 0:
        return None, None
    return parsed.min().isoformat(), parsed.max().isoformat()


def audit_csv(path: str | Path, chunksize: int = DEFAULT_CHUNK_SIZE) -> dict:
    """Audit a CSV incrementally, retaining only small aggregate state."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    total_rows = 0
    missing = Counter()
    buildings: set[int] = set()
    keys: set[tuple[int, str]] = set()
    row_ids: set[int] = set()
    duplicate_keys = 0
    duplicate_row_ids = 0
    label_counts: Counter = Counter()
    start: str | None = None
    end: str | None = None
    columns: list[str] | None = None
    dtypes: dict[str, str] = {}

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if columns is None:
            columns = list(chunk.columns)
            dtypes = {name: str(dtype) for name, dtype in chunk.dtypes.items()}
        total_rows += len(chunk)
        missing.update(chunk.isna().sum().to_dict())
        if "building_id" in chunk:
            buildings.update(chunk["building_id"].dropna().astype(int).tolist())
        if "timestamp" in chunk:
            chunk_start, chunk_end = _timestamp_range(chunk["timestamp"])
            if chunk_start and (start is None or chunk_start < start):
                start = chunk_start
            if chunk_end and (end is None or chunk_end > end):
                end = chunk_end
        if TARGET_COLUMN in chunk:
            label_counts.update(chunk[TARGET_COLUMN].value_counts(dropna=False).to_dict())
        if set(KEY_COLUMNS).issubset(chunk.columns):
            for building, timestamp in zip(
                chunk["building_id"], chunk["timestamp"], strict=True
            ):
                key = (int(building), str(timestamp))
                duplicate_keys += key in keys
                keys.add(key)
        if "row_id" in chunk:
            for row_id in chunk["row_id"]:
                duplicate_row_ids += row_id in row_ids
                row_ids.add(row_id)

    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "columns": columns or [],
        "dtypes": dtypes,
        "row_count": total_rows,
        "missing_values": {k: int(v) for k, v in missing.items() if v},
        "building_count": len(buildings),
        "timestamp_range": {"min": start, "max": end},
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "duplicate_key_count": int(duplicate_keys),
        "duplicate_row_id_count": int(duplicate_row_ids),
    }


def audit_train_buildings(
    path: str | Path, chunksize: int = DEFAULT_CHUNK_SIZE
) -> dict:
    """Return per-building counts and whether each building has both labels."""
    counts: dict[int, Counter] = {}
    readings: Counter = Counter()
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        readings.update(chunk.groupby("building_id")["meter_reading"].count().to_dict())
        for building, labels in chunk.groupby("building_id")["anomaly"]:
            counts.setdefault(int(building), Counter()).update(
                labels.value_counts().to_dict()
            )
    profiles = {
        str(building): {
            "readings": int(readings[building]),
            "label_counts": {str(k): int(v) for k, v in counter.items()},
            "has_both_labels": bool(counter.get(0, 0) and counter.get(1, 0)),
        }
        for building, counter in sorted(counts.items())
    }
    return {
        "building_profiles": profiles,
        "buildings_with_both_labels": sum(
            profile["has_both_labels"] for profile in profiles.values()
        ),
        "buildings_only_normal": sum(
            profile["label_counts"].get("1", 0) == 0 for profile in profiles.values()
        ),
        "buildings_only_anomaly": sum(
            profile["label_counts"].get("0", 0) == 0 for profile in profiles.values()
        ),
    }


def audit_lead_directory(
    root: str | Path, chunksize: int = DEFAULT_CHUNK_SIZE
) -> dict:
    """Audit all five LEAD files and flag feature columns requiring review."""
    root = Path(root)
    reports = {
        name: audit_csv(root / name, chunksize=chunksize)
        for name in (
            "train.csv",
            "test.csv",
            "train_features.csv",
            "test_features.csv",
            "sample_submission.csv",
        )
    }
    train_features = reports["train_features.csv"]["columns"]
    base_train = set(reports["train.csv"]["columns"])
    feature_columns = [column for column in train_features if column not in base_train]
    alignment = _compare_feature_keys(
        root / "train.csv", root / "train_features.csv", chunksize
    )
    alignment["test"] = _compare_feature_keys(
        root / "test.csv", root / "test_features.csv", chunksize
    )
    return {
        "files": reports,
        "train_buildings": audit_train_buildings(
            root / "train.csv", chunksize=chunksize
        ),
        "feature_file_review": {
            "base_join_keys": KEY_COLUMNS,
            "key_alignment": alignment,
            "train_feature_columns": feature_columns,
            "target_derived_risk": [
                column for column in feature_columns if column.startswith("gte_")
            ],
            "decision": (
                "Do not merge feature files: gte_* target encodings have unclear "
                "fit scope and may use labels or validation-period data."
            ),
        },
    }


def _compare_feature_keys(
    base_path: Path, feature_path: Path, chunksize: int
) -> dict[str, int | bool]:
    """Compare base/feature keys in lockstep without materializing either file."""
    mismatches = 0
    target_mismatches = 0
    rows = 0
    base_reader = pd.read_csv(base_path, chunksize=chunksize, low_memory=False)
    feature_reader = pd.read_csv(feature_path, chunksize=chunksize, low_memory=False)
    for base, feature in zip_longest(base_reader, feature_reader):
        if base is None or feature is None:
            extra = feature if base is None else base
            mismatches += len(extra)
            continue
        rows += min(len(base), len(feature))
        if len(base) != len(feature):
            mismatches += abs(len(base) - len(feature))
        base_keys = base[KEY_COLUMNS].astype("string").to_numpy()
        feature_keys = feature[KEY_COLUMNS].astype("string").to_numpy()
        mismatches += int(np.any(base_keys != feature_keys, axis=1).sum())
        if len(base) == len(feature) and TARGET_COLUMN in base and TARGET_COLUMN in feature:
            target_mismatches += int(
                (base[TARGET_COLUMN].to_numpy() != feature[TARGET_COLUMN].to_numpy()).sum()
            )
    return {
        "compared_rows": rows,
        "key_mismatch_rows": mismatches,
        "target_mismatch_rows": target_mismatches,
        "one_to_one_in_file_order": mismatches == 0,
    }


def _valid_lag(
    frame: pd.DataFrame, column: str, periods: int, expected_hours: int
) -> pd.Series:
    values = frame.groupby("building_id", sort=False)[column].shift(periods)
    timestamps = frame.groupby("building_id", sort=False)["timestamp"].shift(periods)
    valid = (frame["timestamp"] - timestamps) == pd.Timedelta(hours=expected_hours)
    return values.where(valid)


def engineer_lead_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build features using only the current and strictly earlier readings."""
    required = {"building_id", "timestamp", "meter_reading"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["meter_reading"] = pd.to_numeric(frame["meter_reading"], errors="coerce")
    frame = frame.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    reading = frame["meter_reading"].where(frame["meter_reading"] >= 0)
    frame["reading_missing"] = reading.isna().astype("int8")
    frame["log_meter_reading"] = np.log1p(reading)
    frame["hour"] = frame["timestamp"].dt.hour.astype("int8")
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek.astype("int8")
    frame["month"] = frame["timestamp"].dt.month.astype("int8")
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)

    for periods, hours in ((1, 1), (24, 24), (168, 168)):
        lag = _valid_lag(frame, "meter_reading", periods, hours)
        frame[f"lag_{periods}"] = lag
        frame[f"diff_{periods}"] = reading - lag
        frame[f"ratio_{periods}"] = (reading / lag.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )

    past = (
        frame.groupby("building_id", sort=False)["meter_reading"]
        .transform(lambda values: values.shift(1).rolling(24, min_periods=1).median())
    )
    frame["past_24_median"] = past
    frame["deviation_from_past_24"] = reading - past
    frame["deviation_ratio_from_past_24"] = (
        reading / past.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    feature_columns = [
        column
        for column in frame.columns
        if column not in {"building_id", "timestamp", TARGET_COLUMN, "row_id"}
    ]
    return frame, feature_columns


def split_by_building(
    frame: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint train/validation row indices with both validation labels."""
    if TARGET_COLUMN not in frame:
        raise ValueError("Building split requires the anomaly target")
    splitter = GroupShuffleSplit(
        n_splits=100, test_size=test_size, random_state=random_state
    )
    groups = frame["building_id"].to_numpy()
    labels = frame[TARGET_COLUMN].to_numpy()
    for train_indices, validation_indices in splitter.split(frame, labels, groups):
        if np.unique(labels[validation_indices]).size == 2:
            return train_indices, validation_indices
    raise ValueError("Could not find a building holdout containing both classes")


def split_validation_buildings(
    frame: pd.DataFrame, *, test_size: float = 0.5, random_state: int = 42
) -> tuple[list[int], list[int]]:
    """Split an existing building holdout into threshold and final groups."""
    if TARGET_COLUMN not in frame or "building_id" not in frame:
        raise ValueError("Validation building split requires building_id and anomaly")
    buildings = sorted(frame["building_id"].unique().tolist())
    threshold, final = train_test_split(
        buildings, test_size=test_size, random_state=random_state
    )
    threshold = sorted(int(building) for building in threshold)
    final = sorted(int(building) for building in final)
    if set(threshold) & set(final):
        raise AssertionError("Threshold and final buildings overlap")
    for group in (threshold, final):
        if frame.loc[frame["building_id"].isin(group), TARGET_COLUMN].nunique() < 2:
            raise ValueError("Each evaluation building group must contain both classes")
    return threshold, final


def sample_rows_by_building(
    frame: pd.DataFrame,
    buildings: Iterable[int],
    *,
    rows_per_building: int = 125,
    random_state: int = 42,
) -> np.ndarray:
    """Sample the same number of labelled rows from every selected building."""
    if rows_per_building < 1:
        raise ValueError("rows_per_building must be positive")
    sampled: list[int] = []
    for offset, building in enumerate(sorted(buildings)):
        candidates = frame.index[frame["building_id"] == building]
        if len(candidates) < rows_per_building:
            raise ValueError(f"Building {building} has too few rows to sample")
        sampled.extend(
            frame.loc[candidates].sample(
                n=rows_per_building, random_state=random_state + offset
            ).index.tolist()
        )
    return np.asarray(sorted(sampled), dtype=int)


def fit_baseline(
    x_train: pd.DataFrame, y_train: pd.Series
) -> SGDClassifier:
    """Fit a reproducible, class-balanced logistic baseline."""
    model = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=100,
        tol=1e-3,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def classification_metrics(
    y_true: Iterable[int], probabilities: Iterable[float], threshold: float
) -> dict:
    y_true_array = np.asarray(list(y_true))
    probability_array = np.asarray(list(probabilities), dtype=float)
    predictions = probability_array >= threshold
    return {
        "roc_auc": float(roc_auc_score(y_true_array, probability_array)),
        "precision": float(precision_score(y_true_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_true_array, predictions, zero_division=0)),
        "f1": float(f1_score(y_true_array, predictions, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true_array, predictions).tolist(),
        "rows": int(len(y_true_array)),
        "anomalies": int(y_true_array.sum()),
    }


def best_f1_threshold(
    y_true: Iterable[int], probabilities: Iterable[float]
) -> float:
    """Select the threshold paired with the best precision-recall operating point."""
    labels = np.asarray(list(y_true))
    scores = np.asarray(list(probabilities), dtype=float)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return 0.5
    f1_values = 2 * precision * recall / np.where(
        precision + recall,
        precision + recall,
        1,
    )
    index = int(np.nanargmax(f1_values))
    return float(thresholds[min(index, len(thresholds) - 1)])


def validate_prediction_alignment(
    baseline: pd.DataFrame, tabpfn: pd.DataFrame
) -> None:
    """Require identical keys and labels before comparing ranked predictions."""
    key_columns = ["row_index", "building_id", "timestamp", TARGET_COLUMN]
    if list(baseline.columns) == [] or list(tabpfn.columns) == []:
        raise ValueError("Prediction artifacts must not be empty")
    for column in key_columns:
        if column not in baseline or column not in tabpfn:
            raise ValueError(f"Prediction artifacts are missing {column}")
    left = baseline[key_columns].reset_index(drop=True)
    right = tabpfn[key_columns].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Prediction artifacts do not have identical row keys and labels")


def ranked_alert_metrics(
    frame: pd.DataFrame,
    score_column: str,
    budgets: Iterable[int] = (25, 50, 100),
) -> dict:
    """Calculate AP and fixed-budget ranked-alert metrics with deterministic ties."""
    required = {"row_index", "building_id", "timestamp", TARGET_COLUMN, score_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing ranked-alert columns: {', '.join(sorted(missing))}")
    if frame[score_column].isna().any():
        raise ValueError("Ranked-alert scores must be finite and non-missing")
    ranked = frame.sort_values(
        [score_column, "building_id", "timestamp", "row_index"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    total_anomalies = int(frame[TARGET_COLUMN].sum())
    result = {
        "average_precision": float(
            average_precision_score(frame[TARGET_COLUMN], frame[score_column])
        ),
        "rows": int(len(frame)),
        "anomalies": total_anomalies,
        "prevalence": float(total_anomalies / len(frame)),
        "alerts": {},
    }
    for budget in budgets:
        if budget < 1 or budget > len(ranked):
            raise ValueError(f"Alert budget must be between 1 and {len(ranked)}")
        top = ranked.head(budget)
        hits = int(top[TARGET_COLUMN].sum())
        result["alerts"][str(budget)] = {
            "k": budget,
            "anomalies_found": hits,
            "precision": float(hits / budget),
            "recall": float(hits / total_anomalies) if total_anomalies else 0.0,
            "row_indices": top["row_index"].astype(int).tolist(),
        }
    return result
