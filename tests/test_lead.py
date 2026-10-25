import numpy as np
import pandas as pd

from meterminder.lead import (
    engineer_lead_features,
    best_f1_threshold,
    classification_metrics,
    ranked_alert_metrics,
    validate_prediction_alignment,
    sample_rows_by_building,
    split_by_building,
    split_validation_buildings,
)


def test_features_sort_rows_and_do_not_use_current_or_future_readings() -> None:
    raw = pd.DataFrame(
        {
            "building_id": [2, 1, 1, 1],
            "timestamp": [
                "2016-01-01 01:00",
                "2016-01-01 02:00",
                "2016-01-01 00:00",
                "2016-01-01 03:00",
            ],
            "meter_reading": [20.0, 30.0, 10.0, 40.0],
            "anomaly": [0, 0, 0, 1],
        }
    )
    features, feature_columns = engineer_lead_features(raw)

    assert list(features["building_id"]) == [1, 1, 1, 2]
    assert np.isnan(features.loc[0, "lag_1"])
    assert np.isnan(features.loc[1, "lag_1"])
    assert features.loc[2, "lag_1"] == 30
    assert features.loc[2, "past_24_median"] == 20
    assert "building_id" not in feature_columns


def test_features_break_lags_at_timestamp_gaps_and_handle_zero_ratios() -> None:
    raw = pd.DataFrame(
        {
            "building_id": [1, 1, 1],
            "timestamp": ["2016-01-01 00:00", "2016-01-01 02:00", "2016-01-01 03:00"],
            "meter_reading": [0.0, 2.0, np.nan],
        }
    )
    features, _ = engineer_lead_features(raw)

    assert np.isnan(features.loc[1, "lag_1"])
    assert features.loc[2, "lag_1"] == 2
    assert np.isnan(features.loc[1, "ratio_1"])
    assert features.loc[2, "reading_missing"] == 1


def test_building_split_is_disjoint_and_has_both_validation_labels() -> None:
    rows = []
    for building in range(10):
        for label in [0, 1]:
            rows.append({"building_id": building, "anomaly": label})
    frame = pd.DataFrame(rows)
    train, validation = split_by_building(frame, test_size=0.3)

    assert set(frame.iloc[train].building_id).isdisjoint(
        set(frame.iloc[validation].building_id)
    )
    assert set(frame.iloc[validation].anomaly) == {0, 1}


def test_evaluation_groups_and_samples_are_building_disjoint() -> None:
    frame = pd.DataFrame(
        [
            {"building_id": building, "anomaly": label, "value": row}
            for building in range(8)
            for row, label in enumerate([0, 0, 1, 0])
        ]
    )
    threshold, final = split_validation_buildings(frame, random_state=42)
    threshold_rows = sample_rows_by_building(frame, threshold, rows_per_building=2)
    final_rows = sample_rows_by_building(frame, final, rows_per_building=2)

    assert set(threshold).isdisjoint(final)
    assert set(frame.loc[threshold_rows, "building_id"]).isdisjoint(
        set(frame.loc[final_rows, "building_id"])
    )
    assert len(threshold_rows) == len(threshold) * 2
    assert len(final_rows) == len(final) * 2


def test_threshold_rows_are_not_final_evaluation_rows() -> None:
    frame = pd.DataFrame(
        [
            {"building_id": building, "anomaly": label}
            for building in range(8)
            for label in [0, 1]
        ]
    )
    threshold, final = split_validation_buildings(frame)
    threshold_rows = sample_rows_by_building(frame, threshold, rows_per_building=2)
    final_rows = sample_rows_by_building(frame, final, rows_per_building=2)

    assert set(threshold_rows).isdisjoint(final_rows)
    assert set(frame.loc[threshold_rows, "anomaly"]) == {0, 1}
    assert set(frame.loc[final_rows, "anomaly"]) == {0, 1}


def test_best_f1_threshold_handles_extra_precision_recall_endpoint() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    threshold = best_f1_threshold(labels, probabilities)

    assert threshold == 0.8


def test_classification_metrics_matches_saved_prediction_confusion_matrix() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.8, 0.7, 0.9]

    metrics = classification_metrics(labels, probabilities, threshold=0.75)

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["anomalies"] == 2


def test_ranked_alert_metrics_calculates_fixed_budget_precision_and_recall() -> None:
    frame = pd.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "building_id": [1, 1, 2, 2],
            "timestamp": ["2020-01-01 00:00"] * 4,
            "anomaly": [1, 0, 1, 0],
            "score": [0.9, 0.8, 0.7, 0.1],
        }
    )

    metrics = ranked_alert_metrics(frame, "score", budgets=(1, 2))

    assert metrics["alerts"]["1"]["anomalies_found"] == 1
    assert metrics["alerts"]["1"]["precision"] == 1.0
    assert metrics["alerts"]["1"]["recall"] == 0.5
    assert metrics["alerts"]["2"]["anomalies_found"] == 1


def test_ranked_alert_ties_use_building_timestamp_then_row_index() -> None:
    frame = pd.DataFrame(
        {
            "row_index": [2, 1, 3],
            "building_id": [2, 1, 1],
            "timestamp": ["2020-01-01 00:00"] * 3,
            "anomaly": [1, 0, 0],
            "score": [0.5, 0.5, 0.5],
        }
    )

    metrics = ranked_alert_metrics(frame, "score", budgets=(2,))

    assert metrics["alerts"]["2"]["row_indices"] == [1, 3]


def test_prediction_alignment_rejects_different_rows_or_labels() -> None:
    frame = pd.DataFrame(
        {
            "row_index": [1],
            "building_id": [2],
            "timestamp": ["2020-01-01"],
            "anomaly": [0],
            "score": [0.5],
        }
    )
    different = frame.assign(anomaly=[1])

    try:
        validate_prediction_alignment(frame, different)
    except ValueError as error:
        assert "identical" in str(error)
    else:
        raise AssertionError("Expected mismatched prediction artifacts to fail")
