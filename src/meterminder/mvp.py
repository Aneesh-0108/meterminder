"""Pure data helpers for the local Streamlit LEAD MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from bisect import bisect_left, bisect_right
import math

import numpy as np
import pandas as pd

from meterminder.lead import validate_prediction_alignment

KEY_COLUMNS = ["row_index", "building_id", "timestamp", "anomaly"]
ALERT_BUDGETS = (25, 50, 100)
REQUIRED_UPLOAD_COLUMNS = {"timestamp", "meter_reading"}
DEMO_DAYS = 14
DEMO_MIN_VALID_COVERAGE = 0.90
DEMO_SHORT_DAYS = 7
DEMO_SHORT_MIN_VALID_COVERAGE = 0.90


def validate_energy_csv(data: pd.DataFrame) -> dict[str, Any]:
    """Validate an hourly meter CSV without silently repairing user data."""
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = sorted(REQUIRED_UPLOAD_COLUMNS.difference(data.columns))
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return {"valid": False, "errors": errors, "warnings": warnings}

    timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
    invalid_timestamps = int(timestamps.isna().sum())
    if invalid_timestamps:
        errors.append(f"{invalid_timestamps} timestamp values are invalid.")
    readings = pd.to_numeric(data["meter_reading"], errors="coerce")
    invalid_readings = int(readings.isna().sum())
    if invalid_readings:
        warnings.append(f"{invalid_readings} meter readings are missing or non-numeric.")
    if (readings.dropna() < 0).any():
        errors.append("meter_reading must be non-negative.")
    valid_times = timestamps.dropna()
    quality = data.copy()
    quality["_timestamp"] = timestamps
    quality["_series"] = (
        quality["building_id"].astype("string")
        if "building_id" in quality
        else pd.Series("single series", index=quality.index)
    )
    valid_quality = quality.dropna(subset=["_timestamp"])
    duplicate_key = valid_quality.duplicated(["_series", "_timestamp"], keep=False)
    duplicate_timestamps = int(
        valid_quality.duplicated(["_series", "_timestamp"], keep="first").sum()
    )
    duplicate_series = int(valid_quality.loc[duplicate_key, "_series"].nunique())
    if duplicate_timestamps:
        warnings.append(
            f"{duplicate_timestamps} duplicate readings across {duplicate_series} "
            "time series; duplicates are retained for review."
        )
    missing_hourly_intervals = 0
    irregular_intervals = 0
    gap_series = 0
    for _, group in valid_quality.groupby("_series", sort=False):
        ordered = group["_timestamp"].sort_values().drop_duplicates()
        gaps = ordered.diff().dropna()
        missing_hourly_intervals += int(
            ((gaps / pd.Timedelta(hours=1)).round().clip(lower=1) - 1)
            .clip(lower=0)
            .sum()
        )
        irregular_intervals += int(
            ((gaps != pd.Timedelta(hours=1)) & (gaps < pd.Timedelta(hours=1))).sum()
        )
        if ((gaps > pd.Timedelta(hours=1)) | (gaps < pd.Timedelta(hours=1))).any():
            gap_series += 1
    non_hourly_gaps = missing_hourly_intervals + irregular_intervals
    if missing_hourly_intervals:
        warnings.append(
            f"{missing_hourly_intervals} missing hourly intervals across {gap_series} "
            "time series."
        )
    if irregular_intervals:
        warnings.append(f"{irregular_intervals} irregular sub-hour intervals found.")
    if "building_id" in data:
        if data["building_id"].isna().any():
            warnings.append("Some building_id values are missing.")
    return {
        "valid": not errors and len(data) > 0,
        "errors": errors,
        "warnings": warnings,
        "rows": int(len(data)),
        "timestamp_min": valid_times.min().isoformat() if len(valid_times) else None,
        "timestamp_max": valid_times.max().isoformat() if len(valid_times) else None,
        "missing_readings": invalid_readings,
        "duplicate_timestamps": duplicate_timestamps,
        "duplicate_series": duplicate_series,
        "missing_hourly_intervals": missing_hourly_intervals,
        "irregular_intervals": irregular_intervals,
        "non_hourly_gaps": non_hourly_gaps,
        "units": "Not encoded in CSV; interpreted as supplied meter_reading units.",
    }


def prepare_energy_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a validated upload for descriptive analysis."""
    report = validate_energy_csv(data)
    if not report["valid"]:
        raise ValueError("; ".join(report["errors"]) or "Energy data is empty.")
    result = data.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["meter_reading"] = pd.to_numeric(result["meter_reading"], errors="coerce")
    if "building_id" not in result:
        result["building_id"] = "Uploaded data"
    result["building_id"] = result["building_id"].fillna("Unknown").astype(str)
    return result.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def descriptive_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Calculate descriptive statistics only; this does not produce predictions."""
    readings = data["meter_reading"].dropna()
    return {
        "rows": int(len(data)),
        "buildings": int(data["building_id"].nunique()),
        "total_consumption": float(readings.sum()) if len(readings) else None,
        "average_consumption": float(readings.mean()) if len(readings) else None,
        "peak_reading": float(readings.max()) if len(readings) else None,
        "peak_periods": data.loc[
            data["meter_reading"].nlargest(min(10, len(data))).index,
            ["timestamp", "building_id", "meter_reading"],
        ].sort_values("meter_reading", ascending=False),
    }


def select_building_period(
    data: pd.DataFrame,
    building_id: str | int | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Apply explicit building and inclusive timestamp filters."""
    result = data
    if building_id is not None:
        result = result[result["building_id"].astype(str) == str(building_id)]
    if start is not None:
        result = result[result["timestamp"] >= start]
    if end is not None:
        result = result[result["timestamp"] <= end]
    return result.copy()


def discover_demo_window(
    csv_path: str | Path,
    *,
    days: int = DEMO_DAYS,
    chunksize: int = 200_000,
    minimum_valid_coverage: float = DEMO_MIN_VALID_COVERAGE,
) -> dict[str, Any]:
    """Find a deterministic window with at least 90% usable readings.

    Windows below the configured coverage threshold are rejected rather than
    displaying missing readings as zero-valued consumption.
    """
    if not 0 < minimum_valid_coverage <= 1:
        raise ValueError("minimum_valid_coverage must be between 0 and 1")
    readings_by_building: dict[int, dict[pd.Timestamp, list[int]]] = {}
    for chunk in pd.read_csv(
        csv_path,
        usecols=["building_id", "timestamp", "meter_reading"],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        numeric = pd.to_numeric(chunk["meter_reading"], errors="coerce")
        for building, timestamp, raw, value in zip(
            chunk["building_id"], chunk["timestamp"], chunk["meter_reading"], numeric,
            strict=True,
        ):
            if pd.isna(timestamp):
                continue
            status = readings_by_building.setdefault(int(building), {}).setdefault(
                timestamp, [0, 0, 0]
            )
            if pd.isna(raw):
                status[1] += 1
            elif pd.isna(value) or not np.isfinite(value) or value < 0:
                status[2] += 1
            else:
                status[0] += 1
    def find_candidates(window_days: int, coverage_threshold: float, fallback: bool) -> list[dict[str, Any]]:
        expected = window_days * 24
        minimum_valid = math.ceil(expected * coverage_threshold)
        candidates: list[dict[str, Any]] = []
        for building, readings in readings_by_building.items():
            timestamps = sorted(readings)
            if not timestamps:
                continue
            valid_counts = np.array([readings[timestamp][0] for timestamp in timestamps])
            missing_counts = np.array([readings[timestamp][1] for timestamp in timestamps])
            invalid_counts = np.array([readings[timestamp][2] for timestamp in timestamps])
            prefix_valid = np.concatenate(([0], np.cumsum(valid_counts)))
            prefix_missing = np.concatenate(([0], np.cumsum(missing_counts)))
            prefix_invalid = np.concatenate(([0], np.cumsum(invalid_counts)))
            prefix_irregular = np.concatenate(
                ([0], np.cumsum([
                    int(current - previous < pd.Timedelta(hours=1))
                    for previous, current in zip(timestamps, timestamps[1:])
                ]))
            )
            first = timestamps[0].floor("h")
            last = timestamps[-1].floor("h") - pd.Timedelta(hours=expected - 1)
            window = first
            while window <= last:
                window_end = window + pd.Timedelta(hours=expected - 1)
                left = bisect_left(timestamps, window)
                right = bisect_right(timestamps, window_end)
                valid = int(prefix_valid[right] - prefix_valid[left])
                missing = int(prefix_missing[right] - prefix_missing[left])
                invalid = int(prefix_invalid[right] - prefix_invalid[left])
                actual = right - left
                expected_missing = expected - actual
                irregular = int(prefix_irregular[right - 1] - prefix_irregular[left])
                if valid >= minimum_valid:
                    candidates.append(
                        {
                            "building_id": building,
                            "start": window,
                            "end": window_end,
                            "expected_rows": expected,
                            "actual_unique_timestamps": actual,
                            "valid_readings": valid,
                            "missing_readings": missing,
                            "invalid_readings": invalid,
                            "missing_hourly_intervals": expected_missing,
                            "irregular_intervals": irregular,
                            "valid_coverage": min(valid / expected, 1.0),
                            "fallback": fallback,
                            "_continuity_penalty": expected_missing + irregular,
                        }
                    )
                window += pd.Timedelta(hours=24)
        return candidates

    candidates = find_candidates(days, minimum_valid_coverage, False)
    if not candidates and days == DEMO_DAYS:
        candidates = find_candidates(
            DEMO_SHORT_DAYS, DEMO_SHORT_MIN_VALID_COVERAGE, True
        )
    if not candidates:
        minimum_valid = math.ceil(days * 24 * minimum_valid_coverage)
        raise ValueError(
            f"No {days}-day window meets the minimum valid-reading coverage of "
            f"{minimum_valid_coverage:.0%} ({minimum_valid} valid readings required)."
        )
    selected = min(
        candidates,
        key=lambda item: (
            -item["valid_coverage"],
            item["_continuity_penalty"],
            item["start"],
            item["building_id"],
        ),
    )
    reason = "highest valid-reading coverage among eligible windows"
    if selected["fallback"]:
        reason = (
            f"no eligible {days}-day window; selected the highest-coverage "
            f"{DEMO_SHORT_DAYS}-day fallback"
        )
    return selected | {"selection_reason": reason}


def load_demo_window(
    csv_path: str | Path, selection: dict[str, Any], chunksize: int = 200_000
) -> pd.DataFrame:
    """Load only the selected building/window from a CSV in a second pass."""
    matches: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=["building_id", "timestamp", "meter_reading"],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        selected = chunk[
            (chunk["building_id"] == selection["building_id"])
            & (chunk["timestamp"] >= selection["start"])
            & (chunk["timestamp"] <= selection["end"])
        ]
        if len(selected):
            matches.append(selected)
    if not matches:
        raise ValueError("The selected demo window could not be loaded.")
    return pd.concat(matches, ignore_index=True)


def load_prediction_artifacts(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    baseline = pd.read_csv(root / "corrected_baseline_predictions.csv")
    tabpfn = pd.read_csv(root / "corrected_tabpfn_predictions.csv")
    if not baseline[KEY_COLUMNS].reset_index(drop=True).equals(
        tabpfn[KEY_COLUMNS].reset_index(drop=True)
    ):
        raise ValueError("Baseline and TabPFN artifacts do not share identical rows")
    baseline["timestamp"] = pd.to_datetime(baseline["timestamp"], errors="coerce")
    tabpfn["timestamp"] = pd.to_datetime(tabpfn["timestamp"], errors="coerce")
    if baseline["timestamp"].isna().any() or tabpfn["timestamp"].isna().any():
        raise ValueError("Prediction artifacts contain invalid timestamps")
    return baseline, tabpfn


def ranked_alerts(frame: pd.DataFrame, budget: int) -> pd.DataFrame:
    if budget not in ALERT_BUDGETS:
        raise ValueError(f"Budget must be one of {ALERT_BUDGETS}")
    return (
        frame.sort_values(
            ["tabpfn_probability", "building_id", "timestamp", "row_index"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        .head(budget)
        .reset_index(drop=True)
    )


def filter_alerts(
    frame: pd.DataFrame,
    buildings: list[int] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    result = frame
    if buildings:
        result = result[result["building_id"].isin(buildings)]
    if start is not None:
        result = result[result["timestamp"] >= start]
    if end is not None:
        result = result[result["timestamp"] <= end]
    return result.copy()


def lookup_readings(
    raw_path: str | Path, keys: pd.DataFrame, chunksize: int = 200_000
) -> pd.DataFrame:
    """Read only the raw train file, in chunks, for displayed keys."""
    wanted = set(zip(keys["building_id"], keys["timestamp"].astype(str)))
    matches: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        raw_path,
        usecols=["building_id", "timestamp", "meter_reading"],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        mask = [
            (building, str(timestamp)) in wanted
            for building, timestamp in zip(
                chunk["building_id"], chunk["timestamp"], strict=True
            )
        ]
        if any(mask):
            matches.append(chunk.loc[mask])
    if not matches:
        return pd.DataFrame(columns=["building_id", "timestamp", "meter_reading"])
    return pd.concat(matches, ignore_index=True)


def lookup_reading_context(
    raw_path: str | Path,
    building_id: int,
    timestamp: pd.Timestamp,
    hours: int = 3,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Read a small timestamp neighborhood for one building in chunks."""
    start = timestamp - pd.Timedelta(hours=hours)
    end = timestamp + pd.Timedelta(hours=hours)
    matches: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        raw_path,
        usecols=["building_id", "timestamp", "meter_reading"],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        selected = chunk[
            (chunk["building_id"] == building_id)
            & (chunk["timestamp"] >= start)
            & (chunk["timestamp"] <= end)
        ]
        if len(selected):
            matches.append(selected)
    if not matches:
        return pd.DataFrame(columns=["building_id", "timestamp", "meter_reading"])
    return pd.concat(matches, ignore_index=True).sort_values("timestamp")


def evaluation_summary(metrics: dict, ranked_metrics: dict) -> pd.DataFrame:
    rows = []
    for model, label in (("baseline", "Baseline"), ("tabpfn", "TabPFN")):
        corrected = metrics["corrected_evaluation"][model]["metrics"]
        ranked = ranked_metrics[model]
        rows.append(
            {
                "Model": label,
                "ROC-AUC": corrected["roc_auc"],
                "Average Precision": ranked["average_precision"],
                "Precision@25": ranked["alerts"]["25"]["precision"],
                "Recall@25": ranked["alerts"]["25"]["recall"],
                "Precision@50": ranked["alerts"]["50"]["precision"],
                "Recall@50": ranked["alerts"]["50"]["recall"],
                "Precision@100": ranked["alerts"]["100"]["precision"],
                "Recall@100": ranked["alerts"]["100"]["recall"],
            }
        )
    return pd.DataFrame(rows).set_index("Model")
