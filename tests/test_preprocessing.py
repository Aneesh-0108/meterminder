from pathlib import Path

import pandas as pd

from meterminder.preprocessing import preprocess_household_data


def _write_fixture(path: Path) -> None:
    rows = [
        "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3",
        "01/01/2007;00:00:00;1.0;0.1;230;4;0;0;0",
        "01/01/2007;00:01:00;2.0;0.1;230;4;0;0;0",
        "01/01/2007;00:02:00;?;0.1;230;4;0;0;0",
        "02/01/2007;00:00:00;3.0;0.1;230;4;0;0;0",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def test_preprocessing_preserves_missingness_and_units(tmp_path: Path) -> None:
    source = tmp_path / "household.txt"
    output = tmp_path / "daily.csv"
    _write_fixture(source)

    daily = preprocess_household_data(source, output, min_coverage=0.5)

    assert output.is_file()
    assert list(daily["date"]) == [pd.Timestamp("2007-01-01"), pd.Timestamp("2007-01-02")]
    assert daily.loc[0, "energy_kwh"] == 3 / 60
    assert daily.loc[0, "valid_readings"] == 2
    assert daily.loc[0, "missing_readings"] == 1438
    assert daily.loc[0, "coverage_ratio"] == 2 / 1440
    assert not bool(daily.loc[0, "is_complete"])
    assert daily.loc[1, "energy_kwh"] == 3 / 60


def test_preprocessing_rejects_invalid_coverage(tmp_path: Path) -> None:
    source = tmp_path / "household.txt"
    _write_fixture(source)

    try:
        preprocess_household_data(source, tmp_path / "daily.csv", min_coverage=0)
    except ValueError as error:
        assert "min_coverage" in str(error)
    else:
        raise AssertionError("Expected invalid coverage to raise ValueError")
