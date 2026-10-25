"""Evaluate fixed-budget ranked anomaly alerts from saved final predictions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from meterminder.lead import ranked_alert_metrics, validate_prediction_alignment


def main() -> None:
    root = Path("data/processed/lead")
    baseline = pd.read_csv(root / "corrected_baseline_predictions.csv")
    tabpfn = pd.read_csv(root / "corrected_tabpfn_predictions.csv")
    validate_prediction_alignment(baseline, tabpfn)
    report = {
        "source": {
            "baseline": "corrected_baseline_predictions.csv",
            "tabpfn": "corrected_tabpfn_predictions.csv",
        },
        "alignment": {
            "rows": len(baseline),
            "buildings": int(baseline["building_id"].nunique()),
            "identical_keys_labels": True,
        },
        "random_ranking_reference": {
            "anomalies": int(baseline["anomaly"].sum()),
            "rows": len(baseline),
            "precision": float(baseline["anomaly"].mean()),
        },
        "baseline": ranked_alert_metrics(baseline, "baseline_probability"),
        "tabpfn": ranked_alert_metrics(tabpfn, "tabpfn_probability"),
    }
    (root / "phase2e_ranked_alerts.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
