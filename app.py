"""MeterMinder upload-first Streamlit MVP."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from meterminder.mvp import (  # noqa: E402
    descriptive_summary,
    evaluation_summary,
    load_prediction_artifacts,
    discover_demo_window,
    lookup_readings,
    load_demo_window,
    prepare_energy_data,
    select_building_period,
    validate_energy_csv,
)
from meterminder.tabpfn_inference import (  # noqa: E402
    DEFAULT_MODEL_REFERENCE,
    DEFAULT_SCHEMA_REFERENCE,
    load_model_bundle,
    score_with_tabpfn,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
ARTIFACT_ROOT = ROOT / "data" / "processed" / "lead"
RAW_TRAIN = ROOT / "data" / "raw" / "lead" / "train.csv"
RAW_TEST = ROOT / "data" / "raw" / "lead" / "test.csv"
TABPFN_MODEL_REFERENCE = ROOT / DEFAULT_MODEL_REFERENCE
TABPFN_SCHEMA_REFERENCE = ROOT / DEFAULT_SCHEMA_REFERENCE


@st.cache_data
def load_benchmark() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    baseline, tabpfn = load_prediction_artifacts(ARTIFACT_ROOT)
    metrics = json.loads((ARTIFACT_ROOT / "metrics.json").read_text(encoding="utf-8"))
    ranked = json.loads(
        (ARTIFACT_ROOT / "phase2e_ranked_alerts.json").read_text(encoding="utf-8")
    )
    return baseline, tabpfn, metrics, ranked


@st.cache_data
def load_demo_data() -> pd.DataFrame:
    if not RAW_TEST.is_file():
        raise FileNotFoundError("Demo data is unavailable because test.csv is missing.")
    selection = discover_demo_window(RAW_TEST)
    data = load_demo_window(RAW_TEST, selection)
    data.attrs["demo_selection"] = selection
    return data


@st.cache_resource
def load_upload_model():
    return load_model_bundle(TABPFN_MODEL_REFERENCE, TABPFN_SCHEMA_REFERENCE)


def safe_model_error(error: Exception) -> str:
    token = os.getenv("TABPFN_TOKEN", "")
    detail = str(error).replace(token, "[REDACTED]") if token else str(error)
    return f"{type(error).__name__}: {detail.replace(chr(10), ' ')}"


def show_quality(report: dict) -> None:
    if report["errors"]:
        for error in report["errors"]:
            st.error(error)
    for warning in report["warnings"]:
        st.warning(warning)
    if report["valid"]:
        st.success(
            f"Validated {report['rows']:,} rows from {report['timestamp_min']} "
            f"to {report['timestamp_max']}."
        )
    st.caption(report["units"])


def analysis_page(data: pd.DataFrame, source: str) -> None:
    st.title("Energy analysis")
    st.caption(f"Source: {source}")
    report = validate_energy_csv(data)
    show_quality(report)
    if not report["valid"]:
        return
    demo_selection = data.attrs.get("demo_selection")
    if demo_selection:
        st.info(
            f"Demo selection: building {demo_selection['building_id']}, "
            f"{demo_selection['start']} to {demo_selection['end']}; "
            f"{demo_selection['actual_unique_timestamps']} of "
            f"{demo_selection['expected_rows']} expected timestamps; "
            f"{demo_selection['valid_readings']} valid readings "
            f"({demo_selection['valid_coverage']:.1%} coverage), "
            f"{demo_selection['missing_readings']} missing and "
            f"{demo_selection['invalid_readings']} invalid readings. "
            f"Missing hourly intervals: {demo_selection['missing_hourly_intervals']}; "
            f"irregular intervals: {demo_selection['irregular_intervals']}. "
            f"{demo_selection['selection_reason']}."
        )
    prepared = prepare_energy_data(data)
    buildings = sorted(prepared["building_id"].unique().tolist())
    selected_building = st.selectbox(
        "Building scope",
        ["All buildings"] + buildings,
        help="Choose one building for facility-level statistics, or keep all buildings for an aggregate overview.",
    )
    min_time = prepared["timestamp"].min()
    max_time = prepared["timestamp"].max()
    selected_dates = st.date_input(
        "Analysis period",
        (min_time.date(), max_time.date()),
        min_value=min_time.date(),
        max_value=max_time.date(),
    )
    start = pd.Timestamp(selected_dates[0])
    end = pd.Timestamp(selected_dates[-1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    scoped = select_building_period(
        prepared,
        None if selected_building == "All buildings" else selected_building,
        start,
        end,
    )
    unit = st.selectbox(
        "Meter-reading units",
        ["Unknown / supplied units", "kWh", "kW"],
        help="LEAD meter_reading units are not declared in the CSV; do not assume kWh.",
    )
    summary = descriptive_summary(scoped)
    st.caption(
        f"Scope: {selected_building} · {start} to {end} · "
        f"units: {unit}"
    )
    cols = st.columns(4)
    cols[0].metric("Rows", f"{summary['rows']:,}")
    cols[1].metric("Buildings", summary["buildings"])
    cols[2].metric(
        f"Total ({unit})",
        "Unavailable" if summary["total_consumption"] is None else f"{summary['total_consumption']:,.2f}",
    )
    cols[3].metric(
        f"Average ({unit})",
        "Unavailable"
        if summary["average_consumption"] is None
        else f"{summary['average_consumption']:,.2f}",
    )
    st.subheader("Consumption trend")
    trend = scoped.set_index("timestamp")["meter_reading"].resample("D").mean()
    st.line_chart(trend)
    st.subheader("Peak-usage periods")
    st.dataframe(summary["peak_periods"], use_container_width=True, hide_index=True)
    st.subheader("Data quality")
    st.write(
        {
            "missing_readings": int(scoped["meter_reading"].isna().sum()),
            "duplicate_readings": int(
                validate_energy_csv(scoped)["duplicate_timestamps"]
            ),
            "missing_hourly_intervals": validate_energy_csv(scoped)[
                "missing_hourly_intervals"
            ],
            "date_range": [start.isoformat(), end.isoformat()],
            "coverage": (
                f"{len(scoped.dropna(subset=['meter_reading']))}/"
                f"{summary['rows']} valid readings"
            ),
        }
    )
    st.info(
        "These are descriptive statistics calculated directly from the uploaded "
        "readings. Experimental anomaly scores require the separately fitted "
        "TabPFN model and are not a replacement for these raw measurements."
    )
    st.subheader("Experimental anomaly review")
    st.warning(
        "Generating scores sends compatible uploaded readings and derived features "
        "to Prior Labs for hosted processing. An anomaly score is not proof of "
        "energy waste, equipment failure, or achievable savings."
    )
    if not TABPFN_MODEL_REFERENCE.is_file():
        st.info(
            "No reusable fitted model is configured locally. Descriptive analysis "
            "remains available; run the one-time setup script before scoring."
        )
    elif st.button("Generate experimental anomaly scores"):
        try:
            model, metadata = load_upload_model()
            scored = score_with_tabpfn(
                data,
                model,
                metadata["feature_columns"],
            )
            st.session_state["analysis_scores"] = scored
            st.success(
                f"Scored {scored['tabpfn_probability'].notna().sum():,} eligible "
                "readings. Rows with missing or invalid current readings were not scored."
            )
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
            st.error(f"Could not generate anomaly scores: {safe_model_error(error)}")
    scored = st.session_state.get("analysis_scores")
    if isinstance(scored, pd.DataFrame):
        ranked = (
            scored.dropna(subset=["tabpfn_probability"])
            .sort_values(
                ["tabpfn_probability", "building_id", "timestamp"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .head(100)
            .copy()
        )
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        st.dataframe(
            ranked[
                [
                    "rank",
                    "building_id",
                    "timestamp",
                    "meter_reading",
                    "tabpfn_probability",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        if len(ranked):
            selected_rank = st.selectbox(
                "Inspect ranked reading",
                ranked["rank"].tolist(),
                key="selected_investigation_rank",
            )
            selected = ranked.loc[ranked["rank"] == selected_rank].iloc[0]
            st.write(
                {
                    "timestamp": selected["timestamp"],
                    "building_id": selected["building_id"],
                    "observed_consumption": selected["meter_reading"],
                    "experimental_anomaly_score": selected["tabpfn_probability"],
                }
            )
            context = scored[
                (scored["building_id"] == selected["building_id"])
                & (
                    scored["timestamp"].between(
                        selected["timestamp"] - pd.Timedelta(hours=3),
                        selected["timestamp"] + pd.Timedelta(hours=3),
                    )
                )
            ]
            st.caption("Available neighboring readings")
            st.dataframe(
                context[["timestamp", "meter_reading", "tabpfn_probability"]],
                use_container_width=True,
                hide_index=True,
            )
    st.caption(
        "Unusual usage is a candidate for investigation, not proof of waste, "
        "equipment failure, or savings."
    )


def performance_page(metrics: dict, ranked: dict) -> None:
    st.title("Model Performance")
    st.caption("LEAD held-out evaluation results; not predictions for uploaded data.")
    st.dataframe(evaluation_summary(metrics, ranked), use_container_width=True)
    st.warning(
        "TabPFN has useful global ranking signal, but fixed-budget top-K results "
        "are limited. These results do not establish production readiness or savings."
    )


def landing_page() -> None:
    st.title("MeterMinder — Understand your energy usage")
    st.write(
        "Explore electricity-consumption patterns and identify unusual periods "
        "worth investigating."
    )
    st.subheader("Choose your data")
    uploaded = st.file_uploader(
        "Upload energy data (CSV)",
        type=["csv"],
        help="Hourly CSV with timestamp, meter_reading, and optional building_id.",
    )
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"Could not read this CSV: {error}")
            return
        st.write("Preview")
        st.dataframe(data.head(10), use_container_width=True, hide_index=True)
        report = validate_energy_csv(data)
        show_quality(report)
        if report["valid"] and st.button("Analyze uploaded data", type="primary"):
            st.session_state["analysis_data"] = data
            st.session_state["analysis_source"] = uploaded.name
            st.session_state.pop("analysis_scores", None)
            st.rerun()
    if st.button("Try demo data"):
        try:
            st.session_state["analysis_data"] = load_demo_data()
            st.session_state["analysis_source"] = "LEAD demo sample"
            st.session_state.pop("analysis_scores", None)
            st.rerun()
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
    st.caption(
        "This phase accepts compatible hourly meter data only. Bills and PDFs "
        "are not supported."
    )


def main() -> None:
    st.set_page_config(page_title="MeterMinder", page_icon="⚡", layout="wide")
    try:
        baseline, tabpfn, metrics, ranked = load_benchmark()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Benchmark artifacts are unavailable: {error}")
        return
    del baseline, tabpfn
    if "analysis_data" not in st.session_state:
        landing_page()
        return
    page = st.sidebar.radio("Page", ["Energy Analysis", "Model Performance", "Start over"])
    if page == "Start over":
        st.session_state.pop("analysis_data", None)
        st.session_state.pop("analysis_source", None)
        st.session_state.pop("analysis_scores", None)
        st.rerun()
    elif page == "Energy Analysis":
        analysis_page(
            st.session_state["analysis_data"],
            st.session_state["analysis_source"],
        )
    else:
        performance_page(metrics, ranked)


if __name__ == "__main__":
    main()
