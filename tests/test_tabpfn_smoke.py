import pytest

from meterminder.tabpfn_smoke import run_tabpfn_cpu_smoke_test


def test_tabpfn_smoke_reports_missing_installation_or_valid_output() -> None:
    try:
        result = run_tabpfn_cpu_smoke_test()
    except RuntimeError as error:
        if "not installed" not in str(error) and "license acceptance" not in str(
            error
        ):
            raise
        pytest.skip(str(error))

    assert result.prediction_shape == (2,)
    assert result.predictions_are_finite
