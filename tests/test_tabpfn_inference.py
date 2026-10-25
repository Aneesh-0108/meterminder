import json

import numpy as np
import pandas as pd
import pytest

from meterminder.tabpfn_inference import (
    load_model_bundle,
    prepare_tabpfn_features,
    score_with_tabpfn,
)


def _hourly_frame(buildings=(1,), rows=180):
    return pd.DataFrame(
        {
            "building_id": np.repeat(buildings, rows),
            "timestamp": [
                timestamp
                for building in buildings
                for timestamp in pd.date_range("2024-01-01", periods=rows, freq="h")
            ],
            "meter_reading": np.tile(np.arange(rows, dtype=float), len(buildings)),
        }
    )


def test_feature_schema_is_stable_for_single_and_multi_building_inputs() -> None:
    single, columns, _ = prepare_tabpfn_features(_hourly_frame())
    multi, multi_columns, _ = prepare_tabpfn_features(_hourly_frame((1, 2)))

    assert columns == multi_columns
    assert "building_id" not in columns
    assert len(single) == 180
    assert len(multi) == 360


def test_missing_readings_are_ineligible_but_not_zero_filled() -> None:
    frame = _hourly_frame(rows=180)
    frame.loc[10, "meter_reading"] = np.nan

    features, _, eligible = prepare_tabpfn_features(frame)

    assert not eligible.iloc[10]
    assert pd.isna(features.iloc[10]["meter_reading"])


def test_lag_warmup_rows_remain_eligible_when_current_reading_is_valid() -> None:
    features, columns, eligible = prepare_tabpfn_features(_hourly_frame(rows=180))

    assert eligible.iloc[0]
    assert pd.isna(features.iloc[0]["lag_1"])
    assert "lag_1" in columns


def test_scores_use_positive_class_one_and_preserve_unscored_rows() -> None:
    class FakeModel:
        classes_ = np.array([1, 0])

        def predict_proba(self, values):
            assert list(values.columns)
            return np.tile([[0.8, 0.2]], (len(values), 1))

    frame = _hourly_frame(rows=180)
    frame.loc[0, "meter_reading"] = np.nan
    features, columns, _ = prepare_tabpfn_features(frame)

    scored = score_with_tabpfn(frame, FakeModel(), columns)

    assert pd.isna(scored.loc[0, "tabpfn_probability"])
    assert scored["tabpfn_probability"].iloc[1] == 0.8
    assert scored["tabpfn_probability"].iloc[1:].notna().all()


def test_invalid_probability_shape_and_nonfinite_scores_fail() -> None:
    class BadModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, values):
            return np.full((len(values), 2), np.nan)

    frame = _hourly_frame(rows=180)
    _, columns, _ = prepare_tabpfn_features(frame)

    with pytest.raises(ValueError, match="non-finite"):
        score_with_tabpfn(frame, BadModel(), columns)


def test_loading_model_reference_does_not_fit(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "model.json"
    schema_path = tmp_path / "schema.json"
    model_path.write_text(json.dumps({"model_id": "server-id"}), encoding="utf-8")
    schema_path.write_text(
        json.dumps(
            {
                "model_identifier": "v3.5_default",
                "feature_columns": ["log_meter_reading"],
            }
        ),
        encoding="utf-8",
    )

    class FakeClassifier:
        @classmethod
        def load_model(cls, source):
            assert source == {"model_id": "server-id"}
            return "loaded-model"

    monkeypatch.setitem(
        __import__("sys").modules,
        "tabpfn_client",
        type("FakeModule", (), {"TabPFNClassifier": FakeClassifier}),
    )

    model, metadata = load_model_bundle(model_path, schema_path)

    assert model == "loaded-model"
    assert metadata["feature_columns"] == ["log_meter_reading"]
