from pathlib import Path

import pandas as pd
import pytest

from meterminder.mvp import (
    descriptive_summary,
    discover_demo_window,
    filter_alerts,
    load_demo_window,
    load_prediction_artifacts,
    prepare_energy_data,
    ranked_alerts,
    validate_energy_csv,
    validate_prediction_alignment,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": [2, 1, 3],
            "building_id": [2, 1, 1],
            "timestamp": pd.to_datetime(
                ["2020-01-02", "2020-01-01", "2020-01-03"]
            ),
            "anomaly": [0, 1, 0],
            "tabpfn_probability": [0.5, 0.5, 0.2],
        }
    )


def test_ranked_alerts_sort_scores_and_ties_deterministically() -> None:
    assert ranked_alerts(_predictions(), 25)["row_index"].tolist() == [1, 2, 3]


def test_filter_alerts_applies_building_and_timestamp_filters() -> None:
    frame = _predictions()
    filtered = filter_alerts(
        frame,
        buildings=[1],
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-02"),
    )
    assert filtered["row_index"].tolist() == [1]


def test_alignment_rejects_different_labels() -> None:
    frame = _predictions()
    with pytest.raises(ValueError, match="identical"):
        validate_prediction_alignment(frame, frame.assign(anomaly=[1, 1, 0]))


def test_load_prediction_artifacts_reads_identical_saved_rows() -> None:
    root = Path("data/processed/lead")
    baseline, tabpfn = load_prediction_artifacts(root)
    assert len(baseline) == len(tabpfn) == 2500
    assert baseline[["row_index", "building_id", "timestamp", "anomaly"]].equals(
        tabpfn[["row_index", "building_id", "timestamp", "anomaly"]]
    )


def test_validate_energy_csv_accepts_hourly_data_and_reports_quality() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "meter_reading": [1.0, 2.0, 3.0],
            "building_id": [1, 1, 1],
        }
    )

    report = validate_energy_csv(frame)

    assert report["valid"]
    assert report["missing_readings"] == 0
    assert report["non_hourly_gaps"] == 0


def test_validate_energy_csv_rejects_missing_malformed_and_empty_input() -> None:
    missing = validate_energy_csv(pd.DataFrame({"meter_reading": [1]}))
    malformed = validate_energy_csv(
        pd.DataFrame({"timestamp": ["not-a-time"], "meter_reading": ["bad"]})
    )
    empty = validate_energy_csv(pd.DataFrame(columns=["timestamp", "meter_reading"]))

    assert not missing["valid"]
    assert not malformed["valid"]
    assert not empty["valid"]
    assert malformed["errors"]


def test_descriptive_summary_does_not_create_predictions() -> None:
    frame = prepare_energy_data(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="h"),
                "meter_reading": [4.0, 6.0],
            }
        )
    )

    summary = descriptive_summary(frame)

    assert summary["total_consumption"] == 10
    assert summary["average_consumption"] == 5
    assert "anomaly" not in summary


def test_validation_uses_composite_building_timestamp_keys() -> None:
    base = pd.Timestamp("2024-01-01")
    valid = pd.DataFrame(
        {
            "building_id": [1, 2],
            "timestamp": [base, base],
            "meter_reading": [1.0, 2.0],
        }
    )
    duplicate = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)

    assert validate_energy_csv(valid)["duplicate_timestamps"] == 0
    assert validate_energy_csv(duplicate)["duplicate_timestamps"] == 1


def test_validation_distinguishes_missing_reading_and_missing_hour() -> None:
    frame = pd.DataFrame(
        {
            "building_id": [1, 1, 1],
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"]
            ),
            "meter_reading": [1.0, None, 3.0],
        }
    )

    report = validate_energy_csv(frame)

    assert report["missing_readings"] == 1
    assert report["missing_hourly_intervals"] == 1
    assert report["duplicate_timestamps"] == 0


def test_single_series_duplicate_timestamps_are_detected() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00", "2024-01-01 00:00"],
            "meter_reading": [1.0, 2.0],
        }
    )

    assert validate_energy_csv(frame)["duplicate_timestamps"] == 1


def test_validation_checks_gaps_per_building_and_unsorted_rows() -> None:
    frame = pd.DataFrame(
        {
            "building_id": [2, 1, 2, 1],
            "timestamp": pd.to_datetime(
                ["2024-01-01 01:00", "2024-01-01 01:00",
                 "2024-01-01 00:00", "2024-01-01 00:00"]
            ),
            "meter_reading": [2.0, 1.0, 2.5, 1.5],
        }
    )

    report = validate_energy_csv(frame)

    assert report["duplicate_timestamps"] == 0
    assert report["missing_hourly_intervals"] == 0


def test_demo_window_selection_is_deterministic_and_loads_selected_rows(tmp_path) -> None:
    path = tmp_path / "test.csv"
    timestamps = pd.date_range("2024-01-01", periods=14 * 24, freq="h")
    pd.DataFrame(
        {
            "row_id": range(len(timestamps)),
            "building_id": [7] * len(timestamps),
            "timestamp": timestamps,
            "meter_reading": [1.0] * len(timestamps),
        }
    ).to_csv(path, index=False)

    first = discover_demo_window(path, chunksize=10)
    second = discover_demo_window(path, chunksize=10)
    loaded = load_demo_window(path, first, chunksize=10)

    assert first == second
    assert first["expected_rows"] == 336
    assert first["actual_unique_timestamps"] == 336
    assert len(loaded) == 336


def _write_demo_fixture(path, windows: list[tuple[int, str, int]]) -> None:
    rows = []
    for building, start, valid_count in windows:
        timestamps = pd.date_range(start, periods=14 * 24, freq="h")
        readings = [1.0 if index < valid_count else None for index in range(len(timestamps))]
        rows.extend(
            {
                "building_id": building,
                "timestamp": timestamp,
                "meter_reading": reading,
            }
            for timestamp, reading in zip(timestamps, readings, strict=True)
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_demo_rejects_complete_timestamp_window_with_all_readings_missing(tmp_path) -> None:
    path = tmp_path / "missing.csv"
    _write_demo_fixture(path, [(1, "2024-01-01", 0)])

    with pytest.raises(ValueError, match="minimum valid-reading coverage"):
        discover_demo_window(path, chunksize=20)


def test_demo_accepts_at_least_90_percent_valid_readings(tmp_path) -> None:
    path = tmp_path / "eligible.csv"
    _write_demo_fixture(path, [(1, "2024-01-01", 303)])

    selection = discover_demo_window(path, chunksize=20)

    assert selection["valid_readings"] == 303
    assert selection["valid_coverage"] >= 0.9
    assert selection["missing_readings"] == 33


def test_demo_prefers_higher_valid_coverage_then_is_deterministic(tmp_path) -> None:
    path = tmp_path / "preferred.csv"
    _write_demo_fixture(
        path,
        [(2, "2024-01-01", 303), (1, "2024-02-01", 330)],
    )

    first = discover_demo_window(path, chunksize=20)
    second = discover_demo_window(path, chunksize=20)

    assert first == second
    assert first["building_id"] == 1
    assert first["valid_readings"] == 330


def test_demo_loader_preserves_missing_readings(tmp_path) -> None:
    path = tmp_path / "preserve.csv"
    _write_demo_fixture(path, [(1, "2024-01-01", 303)])

    selection = discover_demo_window(path, chunksize=20)
    loaded = load_demo_window(path, selection, chunksize=20)

    assert loaded["meter_reading"].isna().sum() == 33


def test_demo_uses_documented_shorter_fallback_when_no_14_day_window_exists(tmp_path) -> None:
    path = tmp_path / "short.csv"
    timestamps = pd.date_range("2024-01-01", periods=7 * 24, freq="h")
    pd.DataFrame(
        {
            "building_id": [1] * len(timestamps),
            "timestamp": timestamps,
            "meter_reading": [1.0] * len(timestamps),
        }
    ).to_csv(path, index=False)

    selection = discover_demo_window(path, chunksize=20)

    assert selection["fallback"] is True
    assert selection["expected_rows"] == 7 * 24
    assert "7-day fallback" in selection["selection_reason"]
