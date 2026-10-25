"""Preprocessing for the UCI Individual Household Electric Power dataset."""

from pathlib import Path

import pandas as pd

UCI_ACTIVE_POWER_COLUMN = "Global_active_power"
DAILY_OUTPUT_PATH = Path("data/processed/daily_consumption.csv")
EXPECTED_MINUTES_PER_DAY = 24 * 60


def preprocess_household_data(
    input_path: str | Path,
    output_path: str | Path = DAILY_OUTPUT_PATH,
    *,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    """Aggregate UCI minute-level active power into daily kWh.

    Missing readings are excluded from the energy sum, never treated as zero.
    Every output day includes its valid-reading count, coverage ratio, missing
    count, and a boolean indicating whether it meets ``min_coverage``.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be greater than 0 and at most 1")

    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    raw = pd.read_csv(
        input_path,
        sep=";",
        na_values=["?"],
        keep_default_na=True,
        low_memory=False,
    )
    required_columns = {"Date", "Time", UCI_ACTIVE_POWER_COLUMN}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {names}")

    timestamps = pd.to_datetime(
        raw["Date"].astype("string") + " " + raw["Time"].astype("string"),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    active_power_kw = pd.to_numeric(raw[UCI_ACTIVE_POWER_COLUMN], errors="coerce")
    valid = timestamps.notna() & active_power_kw.notna()
    if not valid.any():
        raise ValueError("Dataset contains no rows with valid timestamps and power")

    minute_data = pd.DataFrame(
        {
            "timestamp": timestamps[valid],
            "active_power_kw": active_power_kw[valid],
        }
    )
    minute_data["date"] = minute_data["timestamp"].dt.date
    daily = (
        minute_data.groupby("date", as_index=False)
        .agg(
            energy_kwh=("active_power_kw", lambda values: values.sum() / 60),
            valid_readings=("active_power_kw", "size"),
        )
        .assign(
            expected_readings=EXPECTED_MINUTES_PER_DAY,
        )
    )
    daily["missing_readings"] = (
        daily["expected_readings"] - daily["valid_readings"]
    ).clip(lower=0)
    daily["coverage_ratio"] = (
        daily["valid_readings"] / daily["expected_readings"]
    )
    daily["is_complete"] = daily["coverage_ratio"] >= min_coverage
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[
        [
            "date",
            "energy_kwh",
            "valid_readings",
            "expected_readings",
            "missing_readings",
            "coverage_ratio",
            "is_complete",
        ]
    ].sort_values("date", ignore_index=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return daily

