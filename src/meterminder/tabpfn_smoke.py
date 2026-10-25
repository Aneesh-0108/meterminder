"""Small, local-only TabPFN regression smoke test."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SmokeTestResult:
    """Observable result of a TabPFN CPU smoke test."""

    package_version: str
    prediction_shape: tuple[int, ...]
    predictions_are_finite: bool


def run_tabpfn_cpu_smoke_test() -> SmokeTestResult:
    """Fit TabPFN on a tiny synthetic regression dataset using CPU.

    This function intentionally does not use project data or start a service.
    TabPFN downloads model weights on first use, so failures are allowed to
    propagate with their original exception and installation/licensing context.
    """
    try:
        import tabpfn
        from tabpfn import TabPFNRegressor
        from tabpfn.errors import TabPFNLicenseError
    except ImportError as error:
        raise RuntimeError(
            "TabPFN is not installed. Install the pinned project dependencies "
            "before running the CPU smoke test."
        ) from error

    x_train = np.arange(12, dtype=np.float32).reshape(-1, 1)
    y_train = (2 * x_train[:, 0] + 1).astype(np.float32)
    x_test = np.array([[12], [13]], dtype=np.float32)

    model = TabPFNRegressor(device="cpu")
    try:
        model.fit(x_train, y_train)
    except TabPFNLicenseError as error:
        raise RuntimeError(
            "TabPFN model weights require license acceptance before local "
            "inference. See the original TabPFN error for authentication "
            "instructions."
        ) from error
    predictions = np.asarray(model.predict(x_test))
    if predictions.shape != (len(x_test),):
        raise AssertionError(
            f"Unexpected prediction shape: {predictions.shape}; expected "
            f"{(len(x_test),)}"
        )
    return SmokeTestResult(
        package_version=getattr(tabpfn, "__version__", "unknown"),
        prediction_shape=predictions.shape,
        predictions_are_finite=bool(np.isfinite(predictions).all()),
    )
