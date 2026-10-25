"""Fit and persist the one reusable hosted TabPFN upload model.

This script intentionally performs at most one hosted fit. It must be run
manually after checking the account quota; it is never called by pytest or the
Streamlit app.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from meterminder.lead import engineer_lead_features, sample_rows_by_building


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/lead/train.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/lead")
    )
    parser.add_argument("--rows-per-building", type=int, default=100)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)

    from tabpfn_client import get_api_usage
    from tabpfn_client.client import ServiceClient

    print(get_api_usage())
    settings = ServiceClient.get_settings()
    if settings is None:
        raise RuntimeError("TabPFN model availability could not be verified.")
    available = {str(version).split(".")[-1].lower() for version in settings.model_limits}
    if not any("v3_5" in version or "v3.5" in version for version in available):
        raise RuntimeError("The hosted v3.5 TabPFN model is not available.")

    raw = pd.read_csv(args.train, low_memory=False)
    features, feature_columns = engineer_lead_features(raw)
    sampled = sample_rows_by_building(
        features,
        features["building_id"].unique(),
        rows_per_building=args.rows_per_building,
        random_state=42,
    )
    x_train = features.iloc[sampled][feature_columns]
    y_train = features.iloc[sampled]["anomaly"]
    if y_train.nunique() != 2:
        raise ValueError("The deterministic training sample does not contain both classes.")
    model_name = "v3.5_default"
    from tabpfn_client import TabPFNClassifier

    model = TabPFNClassifier(
        model_path=model_name,
        n_estimators=4,
        random_state=42,
        inference_config={"SUBSAMPLE_SAMPLES": len(sampled)},
    )
    model.fit(x_train, y_train, description="MeterMinder upload anomaly scoring")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "tabpfn_upload_model.json"
    schema_path = args.output_dir / "tabpfn_upload_schema.json"
    model_record = model.save_model(model_path)
    schema_path.write_text(
        json.dumps(
            {
                "model_identifier": model_name,
                "feature_columns": feature_columns,
                "training_rows": int(len(sampled)),
                "training_buildings": int(features.iloc[sampled]["building_id"].nunique()),
                "classes": np.asarray(model.classes_).tolist(),
                "tabpfn_client_version": model_record["tabpfn_client_version"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"model_identifier={model_name}")
    print(f"training_rows={len(sampled)}")
    print(f"training_buildings={features.iloc[sampled]['building_id'].nunique()}")
    print(f"model_reference={model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
