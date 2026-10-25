"""Run the reproducible LEAD audit and baseline, with an optional one-shot TabPFN call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer

from meterminder.lead import (
    audit_lead_directory,
    classification_metrics,
    engineer_lead_features,
    fit_baseline,
    sample_rows_by_building,
    split_by_building,
    split_validation_buildings,
)


def _best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    return best_f1_threshold(y_true, probabilities)


def _load_tabpfn():
    from tabpfn_client import TabPFNClassifier

    return TabPFNClassifier


def _run_tabpfn(
    features: pd.DataFrame,
    feature_columns: list[str],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    output_dir: Path,
) -> dict:
    """Make one bounded hosted request after selecting disjoint building subsets."""
    train_rows = train_indices[: min(20_000, len(train_indices))]
    validation_rows = validation_indices[: min(5_000, len(validation_indices))]
    x_train = features.iloc[train_rows][feature_columns]
    y_train = features.iloc[train_rows]["anomaly"]
    x_validation = features.iloc[validation_rows][feature_columns]
    y_validation = features.iloc[validation_rows]["anomaly"]
    if y_train.nunique() < 2 or y_validation.nunique() < 2:
        raise ValueError("TabPFN sample must contain both classes in both partitions")

    classifier_type = _load_tabpfn()
    model_name = "v3.5_default"
    model = classifier_type(
        model_path=model_name,
        n_estimators=4,
        random_state=42,
        inference_config={"SUBSAMPLE_SAMPLES": len(train_rows)},
    )
    model.fit(x_train, y_train)
    probabilities = np.asarray(model.predict_proba(x_validation))[:, 1]
    if probabilities.shape != (len(validation_rows),) or not np.isfinite(probabilities).all():
        raise ValueError("TabPFN returned an invalid probability vector")
    threshold = _best_f1_threshold(y_validation, probabilities)
    predictions = pd.DataFrame(
        {
            "row_index": validation_rows,
            "building_id": features.iloc[validation_rows]["building_id"].to_numpy(),
            "timestamp": features.iloc[validation_rows]["timestamp"].to_numpy(),
            "anomaly": y_validation.to_numpy(),
            "tabpfn_probability": probabilities,
            "tabpfn_prediction": probabilities >= threshold,
        }
    )
    predictions.to_csv(output_dir / "tabpfn_predictions.csv", index=False)
    return {
        "status": "success",
        "model_identifier": model_name,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "metrics": classification_metrics(y_validation, probabilities, threshold),
        "timings": getattr(model, "get_timings", lambda: None)(),
    }


def _run_corrected_tabpfn(
    features: pd.DataFrame,
    feature_columns: list[str],
    train_indices: np.ndarray,
    threshold_rows: np.ndarray,
    final_rows: np.ndarray,
    output_dir: Path,
) -> dict:
    """Make one request for disjoint threshold and final building samples."""
    validation_rows = np.concatenate([threshold_rows, final_rows])
    train_rows = train_indices[: min(20_000, len(train_indices))]
    x_train = features.iloc[train_rows][feature_columns]
    y_train = features.iloc[train_rows]["anomaly"]
    x_validation = features.iloc[validation_rows][feature_columns]
    y_validation = features.iloc[validation_rows]["anomaly"]
    if y_train.nunique() < 2 or y_validation.nunique() < 2:
        raise ValueError("TabPFN samples must contain both classes")

    classifier_type = _load_tabpfn()
    model = classifier_type(
        model_path="v3.5_default",
        n_estimators=4,
        random_state=42,
        inference_config={"SUBSAMPLE_SAMPLES": len(train_rows)},
    )
    model.fit(x_train, y_train)
    probabilities = np.asarray(model.predict_proba(x_validation))[:, 1]
    if probabilities.shape != (len(validation_rows),) or not np.isfinite(probabilities).all():
        raise ValueError("TabPFN returned an invalid probability vector")
    threshold_probability = probabilities[: len(threshold_rows)]
    final_probability = probabilities[len(threshold_rows) :]
    threshold = _best_f1_threshold(
        features.iloc[threshold_rows]["anomaly"], threshold_probability
    )
    predictions = pd.DataFrame(
        {
            "row_index": final_rows,
            "building_id": features.iloc[final_rows]["building_id"].to_numpy(),
            "timestamp": features.iloc[final_rows]["timestamp"].to_numpy(),
            "anomaly": features.iloc[final_rows]["anomaly"].to_numpy(),
            "tabpfn_probability": final_probability,
            "tabpfn_prediction": final_probability >= threshold,
        }
    )
    predictions.to_csv(output_dir / "corrected_tabpfn_predictions.csv", index=False)
    return {
        "status": "success",
        "model_identifier": "v3.5_default",
        "train_rows": len(train_rows),
        "threshold_rows": len(threshold_rows),
        "final_rows": len(final_rows),
        "threshold_buildings": sorted(
            features.iloc[threshold_rows]["building_id"].unique().tolist()
        ),
        "final_buildings": sorted(
            features.iloc[final_rows]["building_id"].unique().tolist()
        ),
        "threshold_class_counts": features.iloc[threshold_rows]["anomaly"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "final_class_counts": features.iloc[final_rows]["anomaly"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "metrics": classification_metrics(
            features.iloc[final_rows]["anomaly"], final_probability, threshold
        ),
        "timings": getattr(model, "get_timings", lambda: None)(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lead"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/lead"))
    parser.add_argument("--run-tabpfn", action="store_true")
    parser.add_argument("--correct-evaluation", action="store_true")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit = audit_lead_directory(args.data_root)
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    raw = pd.read_csv(args.data_root / "train.csv", low_memory=False)
    features, feature_columns = engineer_lead_features(raw)
    train_indices, validation_indices = split_by_building(features)
    x_train = features.iloc[train_indices][feature_columns]
    x_validation = features.iloc[validation_indices][feature_columns]
    y_train = features.iloc[train_indices]["anomaly"]
    y_validation = features.iloc[validation_indices]["anomaly"]

    imputer = SimpleImputer(strategy="median")
    x_train_imputed = imputer.fit_transform(x_train)
    x_validation_imputed = imputer.transform(x_validation)
    baseline = fit_baseline(pd.DataFrame(x_train_imputed), y_train)
    baseline_probabilities = baseline.predict_proba(x_validation_imputed)[:, 1]
    threshold = _best_f1_threshold(y_validation, baseline_probabilities)
    predictions = pd.DataFrame(
        {
            "row_index": validation_indices,
            "building_id": features.iloc[validation_indices]["building_id"].to_numpy(),
            "timestamp": features.iloc[validation_indices]["timestamp"].to_numpy(),
            "anomaly": y_validation.to_numpy(),
            "baseline_probability": baseline_probabilities,
            "baseline_prediction": baseline_probabilities >= threshold,
        }
    )
    predictions.to_csv(args.output_dir / "baseline_predictions.csv", index=False)

    metrics = {
        "feature_columns": feature_columns,
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "train_buildings": int(features.iloc[train_indices]["building_id"].nunique()),
        "validation_buildings": int(
            features.iloc[validation_indices]["building_id"].nunique()
        ),
        "train_class_proportions": y_train.value_counts(normalize=True).sort_index().to_dict(),
        "validation_class_proportions": y_validation.value_counts(normalize=True).sort_index().to_dict(),
        "validation_date_range": [
            features.iloc[validation_indices]["timestamp"].min().isoformat(),
            features.iloc[validation_indices]["timestamp"].max().isoformat(),
        ],
        "baseline": {
            "model": "sklearn.SGDClassifier(log_loss, class_weight=balanced)",
            "metrics": classification_metrics(
                y_validation, baseline_probabilities, threshold
            ),
        },
        "same_validation_sample": {
            "rows": min(5_000, len(validation_indices)),
            "baseline_metrics": classification_metrics(
                y_validation.iloc[: min(5_000, len(validation_indices))],
                baseline_probabilities[: min(5_000, len(validation_indices))],
                _best_f1_threshold(
                    y_validation.iloc[: min(5_000, len(validation_indices))],
                    baseline_probabilities[: min(5_000, len(validation_indices))],
                ),
            ),
        },
        "tabpfn": {"status": "not_run"},
    }
    if args.run_tabpfn:
        metrics["tabpfn"] = _run_tabpfn(
            features,
            feature_columns,
            train_indices,
            validation_indices,
            args.output_dir,
        )
    if args.correct_evaluation:
        threshold_buildings, final_buildings = split_validation_buildings(features.iloc[validation_indices])
        validation_frame = features.iloc[validation_indices].copy()
        threshold_rows = sample_rows_by_building(
            validation_frame, threshold_buildings, rows_per_building=125
        )
        final_rows = sample_rows_by_building(
            validation_frame, final_buildings, rows_per_building=125
        )
        baseline_threshold = _best_f1_threshold(
            features.iloc[threshold_rows]["anomaly"],
            baseline_probabilities[
                np.isin(validation_indices, threshold_rows)
            ],
        )
        baseline_final_probability = baseline_probabilities[
            np.isin(validation_indices, final_rows)
        ]
        corrected_baseline = pd.DataFrame(
            {
                "row_index": final_rows,
                "building_id": features.iloc[final_rows]["building_id"].to_numpy(),
                "timestamp": features.iloc[final_rows]["timestamp"].to_numpy(),
                "anomaly": features.iloc[final_rows]["anomaly"].to_numpy(),
                "baseline_probability": baseline_final_probability,
                "baseline_prediction": baseline_final_probability >= baseline_threshold,
            }
        )
        corrected_baseline.to_csv(
            args.output_dir / "corrected_baseline_predictions.csv", index=False
        )
        threshold_rows_local = np.flatnonzero(np.isin(validation_indices, threshold_rows))
        final_rows_local = np.flatnonzero(np.isin(validation_indices, final_rows))
        corrected = {
            "building_split": {
                "threshold_buildings": threshold_buildings,
                "final_buildings": final_buildings,
                "overlap": sorted(set(threshold_buildings) & set(final_buildings)),
            },
            "threshold_sample": {
                "rows": len(threshold_rows),
                "building_ids": threshold_buildings,
                "row_indices": sorted(map(int, threshold_rows)),
                "class_counts": features.iloc[threshold_rows]["anomaly"]
                .value_counts()
                .sort_index()
                .to_dict(),
            },
            "final_sample": {
                "rows": len(final_rows),
                "building_ids": final_buildings,
                "row_indices": sorted(map(int, final_rows)),
                "class_counts": features.iloc[final_rows]["anomaly"]
                .value_counts()
                .sort_index()
                .to_dict(),
            },
            "baseline": {
                "threshold": baseline_threshold,
                "metrics": classification_metrics(
                    features.iloc[final_rows]["anomaly"],
                    baseline_final_probability,
                    baseline_threshold,
                ),
            },
            "tabpfn": {"status": "not_run"},
        }
        if args.run_tabpfn:
            corrected["tabpfn"] = _run_corrected_tabpfn(
                features,
                feature_columns,
                train_indices,
                threshold_rows,
                final_rows,
                args.output_dir,
            )
        metrics["corrected_evaluation"] = corrected
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
    best_f1_threshold,
