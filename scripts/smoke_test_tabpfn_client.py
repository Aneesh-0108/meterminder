"""Run a minimal authenticated TabPFN Client regression smoke test."""

from pathlib import Path
import os
import sys

import numpy as np
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _safe_error_detail(error: Exception, token: str) -> str:
    """Return useful error text with the credential removed."""
    detail = str(error).replace(token, "[REDACTED]")
    return detail.replace("\r", " ").replace("\n", " ")


def main() -> int:
    """Fit and predict with TabPFN Client without exposing credentials."""
    token = os.getenv("TABPFN_TOKEN")
    if not token:
        print("FAIL: TABPFN_TOKEN is missing or empty in the project .env file.")
        return 1

    try:
        import tabpfn_client
        from tabpfn_client import TabPFNRegressor
    except ImportError as error:
        print(f"FAIL: could not import tabpfn-client: {error}")
        return 1

    print(f"client_version={getattr(tabpfn_client, '__version__', 'unknown')}")
    print("test_row_count=2")

    x_train = np.arange(12, dtype=np.float64).reshape(-1, 1)
    y_train = 2 * x_train[:, 0] + 1
    x_test = np.array([[12.0], [13.0]])

    try:
        estimator = TabPFNRegressor()
        estimator.fit(x_train, y_train)
        predictions = np.asarray(estimator.predict(x_test))
    except Exception as error:
        error_name = type(error).__name__
        print(
            "FAIL: TabPFN Client request failed "
            f"({error_name}). Check authentication, quota, network, and API "
            "availability without sharing the token."
        )
        print(f"detail={_safe_error_detail(error, token)}")
        return 1

    expected_shape = (len(x_test),)
    if predictions.shape != expected_shape:
        print(
            "FAIL: unexpected prediction shape "
            f"{predictions.shape}; expected {expected_shape}."
        )
        return 1
    if not np.issubdtype(predictions.dtype, np.number):
        print(f"FAIL: predictions are not numeric (dtype={predictions.dtype}).")
        return 1
    if not np.isfinite(predictions).all():
        print("FAIL: predictions contain non-finite numeric values.")
        return 1

    print(f"prediction_shape={predictions.shape}")
    print("status=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
