"""Reusable hosted TabPFN inference for compatible hourly meter uploads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from meterminder.lead import engineer_lead_features

DEFAULT_MODEL_REFERENCE = Path("data/processed/lead/tabpfn_upload_model.json")
DEFAULT_SCHEMA_REFERENCE = Path("data/processed/lead/tabpfn_upload_schema.json")


def prepare_tabpfn_features(
    data: pd.DataFrame, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Build the benchmark features and identify rows eligible for scoring."""
    frame = data.copy()
    frame = frame.drop(columns=["anomaly"], errors="ignore")
    if "building_id" not in frame:
        frame["building_id"] = "uploaded-series"
    features, generated_columns = engineer_lead_features(frame)
    columns = generated_columns if feature_columns is None else feature_columns
    missing = sorted(set(columns).difference(features.columns))
    if missing:
        raise ValueError(f"Uploaded data cannot produce feature columns: {missing}")
    readings = pd.to_numeric(features["meter_reading"], errors="coerce")
    eligible = (
        features["timestamp"].notna()
        & readings.notna()
        & np.isfinite(readings)
        & (readings >= 0)
    )
    return features, columns, eligible


def load_model_bundle(
    model_reference: str | Path = DEFAULT_MODEL_REFERENCE,
    schema_reference: str | Path = DEFAULT_SCHEMA_REFERENCE,
) -> tuple[Any, dict[str, Any]]:
    """Load a saved account-bound model record and its feature schema."""
    model_path = Path(model_reference)
    schema_path = Path(schema_reference)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No fitted TabPFN model reference was found at {model_path}."
        )
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"No TabPFN feature schema was found at {schema_path}."
        )
    record = json.loads(model_path.read_text(encoding="utf-8"))
    metadata = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("The saved TabPFN feature schema is missing feature_columns.")
    if metadata.get("model_identifier") != "v3.5_default":
        raise ValueError("The saved model is not the expected v3.5_default model.")
    from tabpfn_client import TabPFNClassifier

    return TabPFNClassifier.load_model(record), metadata


def score_with_tabpfn(
    data: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Return sorted feature rows with class-1 scores for eligible readings."""
    features, columns, eligible = prepare_tabpfn_features(data, feature_columns)
    if columns != feature_columns:
        raise ValueError("Feature columns do not match the saved model schema.")
    result = features.copy()
    result["tabpfn_probability"] = np.nan
    if not eligible.any():
        return result
    probabilities = np.asarray(model.predict_proba(features.loc[eligible, columns]))
    classes = np.asarray(getattr(model, "classes_", []))
    class_one = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or probabilities.shape != (
        int(eligible.sum()),
        len(classes),
    ):
        raise ValueError("TabPFN returned an invalid probability matrix shape.")
    if len(class_one) != 1:
        raise ValueError("TabPFN model does not expose exactly one positive class.")
    scores = probabilities[:, int(class_one[0])]
    if not np.isfinite(scores).all():
        raise ValueError("TabPFN returned non-finite anomaly scores.")
    result.loc[eligible, "tabpfn_probability"] = scores
    return result
