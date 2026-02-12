from __future__ import annotations

import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from column_compare.analysis import (
    compute_peak_metrics,
    extract_chromatograms,
    load_ms1_experiment,
    parse_metabolite_table,
    rank_columns,
    summarise_columns,
)
from column_compare.storage import (
    load_rt_library,
    merge_expected_rts,
    rt_library_to_dataframe,
    save_rt_library,
    update_rt_library_from_metrics,
)

DEFAULT_LIBRARY_PATH = "data/retention_library.json"


def infer_default_column(run_name: str) -> str:
    stem = Path(run_name).stem
    for delimiter in ("__", "_", "-", "."):
        if delimiter in stem:
            candidate = stem.split(delimiter)[0].strip()
            if candidate:
                return candidate
    return stem


def read_metabolite_table(uploaded_file) -> pd.DataFrame:
    raw_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw_bytes))

    try:
        return pd.read_csv(io.BytesIO(raw_bytes))
    except pd.errors.ParserError:
        return pd.read_csv(io.BytesIO(raw_bytes), sep=";")


def default_metabolite_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metabolite": ["Lactate", "Pyruvate", "Citrate", "Glucose"],
            "mz": [89.0244, 87.0088, 191.0197, 179.0556],
            "expected_rt_min": [2.3, 2.9, 4.1, 5.0],
        }
    )


def hash_inputs(
    files: list[Any],
    metabolite_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    tolerance: float,
    tolerance_unit: str,
    rt_window_min: float,
    name_col: str,
    mz_col: str,
    rt_col: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(tolerance).encode())
    digest.update(tolerance_unit.encode())
    digest.update(str(rt_window_min).encode())
    digest.update(name_col.encode())
    digest.update(mz_col.encode())
    digest.update((rt_col or "").encode())
    digest.update(metabolite_df.to_csv(index=False).encode())
    digest.update(mapping_df.to_csv(index=False).encode())
    for uploaded in files:
        digest.update(uploaded.name.encode())
        digest.update(str(uploaded.size).encode())
        digest.update(uploaded.getvalue())
    return digest.hexdigest()


def build_plot_df(chromatograms: dict[tuple[str, str, str], Any], metabolite: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (run_id, column_id, metabolite_name), trace in chromatograms.items():
        if metabolite_name != metabolite:
            continue
        if trace.time_min.size == 0:
            continue
        frame = pd.DataFrame(
            {
                "time_min": trace.time_min,
                "intensity": trace.intensity,
                "run_id": run_id,
                "column_id": column_id,
            }
        )
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["time_min", "intensity", "run_id", "column_id"])
    return pd.concat(rows, ignore_index=True)


def compute_library_fit(library_df: pd.DataFrame, selected_metabolites: list[str]) -> pd.DataFrame:
    if library_df.empty or not selected_metabolites:
        return pd.DataFrame()

    scoped = library_df[library_df["metabolite"].isin(selected_metabolites)].copy()
    if scoped.empty:
        return pd.DataFrame()

    fit = (
        scoped.groupby("column_id")
        .agg(
            metabolites_covered=("metabolite", "nunique"),
            median_rt_std=("std_rt_min", "median"),
            min_observations=("n_observations", "min"),
        )
        .reset_index()
    )
    fit["coverage_fraction"] = fit["metabolites_covered"] / max(len(selected_metabolites), 1)

    std = fit["median_rt_std"].astype(float)
    if std.notna().sum() == 0:
        fit["stability_score"] = 0.0
    else:
        low = float(std.min(skipna=True))
        high = float(std.max(skipna=True))
        if low == high:
            fit["stability_score"] = 1.0
        else:
            fit["stability_score"] = 1.0 - ((std - low) / (high - low)).fillna(0.0)

    fit["library_fit_score"] = 0.75 * fit["coverage_fraction"] + 0.25 * fit["stability_score"]
    return fit.sort_values("library_fit_score", ascending=False).reset_index(drop=True)


def run_analysis(
    uploaded_runs: list[Any],
    mapping_df: pd.DataFrame,
    metabolite_df: pd.DataFrame,
    name_col: str,
    mz_col: str,
    rt_col: str | None,
    tolerance: float,
    tolerance_unit: str,
    rt_window_min: float,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], Any]]:
    targets = parse_metabolite_table(metabolite_df, name_col=name_col, mz_col=mz_col, rt_col=rt_col)
    if not targets:
        raise ValueError("No valid metabolite rows were found after parsing your table.")

    uploaded_by_name = {file.name: file for file in uploaded_runs}
    metrics_rows: list[dict[str, Any]] = []
    chromatograms: dict[tuple[str, str, str], Any] = {}

    for _, row in mapping_df.iterrows():
        file_name = str(row["file_name"])
        run_id = str(row["run_id"]).strip() or Path(file_name).stem
        column_id = str(row["column_id"]).strip() or run_id

        uploaded = uploaded_by_name[file_name]
        with tempfile.NamedTemporaryFile(suffix=".mzXML", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        try:
            exp = load_ms1_experiment(tmp_path)
            trace_map = extract_chromatograms(
                experiment=exp,
                targets=targets,
                tolerance=tolerance,
                tolerance_unit=tolerance_unit,
            )
            for target in targets:
                trace = trace_map[target.name]
                metrics = compute_peak_metrics(
                    chromatogram=trace,
                    run_id=run_id,
                    column_id=column_id,
                    target=target,
                    expected_rt_min=target.expected_rt_min,
                    rt_window_min=rt_window_min,
                )
                metrics_rows.append(metrics.to_record())
                chromatograms[(run_id, column_id, target.name)] = trace
        finally:
            tmp_path.unlink(missing_ok=True)

    return pd.DataFrame(metrics_rows), chromatograms


def app() -> None:
    st.set_page_config(
        page_title="Metabolite Column Comparison",
        page_icon=":test_tube:",
        layout="wide",
    )
    st.title("Metabolite Column Comparison for Standard Runs")
    st.write(
        "Upload multiple mzXML files from different columns and compare extracted peak shapes, "
        "retention times, and efficiency metrics for your metabolite standards."
    )

    with st.sidebar:
        st.header("Analysis Settings")
        tolerance_unit = st.selectbox("m/z tolerance unit", ["ppm", "Da"], index=0)
        tolerance = st.number_input(
            "m/z tolerance",
            min_value=0.00001,
            value=10.0 if tolerance_unit == "ppm" else 0.01,
            step=1.0 if tolerance_unit == "ppm" else 0.001,
            format="%.5f",
        )
        rt_window_min = st.number_input(
            "Expected RT search window (+/- min)",
            min_value=0.0,
            value=0.6,
            step=0.1,
            help="When expected RT is known, peak apex search is constrained to this window.",
        )
        min_snr_for_save = st.number_input(
            "Minimum S/N to save RT standards",
            min_value=0.0,
            value=3.0,
            step=0.5,
        )
        library_path = st.text_input("Retention library file", value=DEFAULT_LIBRARY_PATH)

    rt_library = load_rt_library(library_path)
    library_df = rt_library_to_dataframe(rt_library)

    st.subheader("1) Upload metabolite list")
    metabolite_file = st.file_uploader(
        "Metabolite table (CSV/Excel). Must contain metabolite name and m/z.",
        type=["csv", "txt", "xlsx", "xls"],
        accept_multiple_files=False,
    )

    if metabolite_file is None:
        st.info("No metabolite table uploaded yet. A template is shown below.")
        template_df = default_metabolite_template()
        st.dataframe(template_df, use_container_width=True)
        st.download_button(
            label="Download metabolite template (CSV)",
            data=template_df.to_csv(index=False).encode(),
            file_name="metabolite_template.csv",
            mime="text/csv",
        )
        return

    try:
        metabolite_df = read_metabolite_table(metabolite_file)
    except Exception as exc:
        st.error(f"Failed to read metabolite table: {exc}")
        return

    if metabolite_df.empty:
        st.warning("Uploaded metabolite table is empty.")
        return

    all_columns = list(metabolite_df.columns)
    guessed_name_col = next((c for c in all_columns if "metab" in c.lower()), all_columns[0])
    guessed_mz_col = next((c for c in all_columns if c.lower() in {"mz", "m/z"}), all_columns[0])
    guessed_rt_col = next((c for c in all_columns if "rt" in c.lower()), None)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        name_col = st.selectbox("Metabolite name column", all_columns, index=all_columns.index(guessed_name_col))
    with col_b:
        mz_col = st.selectbox("m/z column", all_columns, index=all_columns.index(guessed_mz_col))
    with col_c:
        rt_choices = ["(none)"] + all_columns
        default_rt_index = rt_choices.index(guessed_rt_col) if guessed_rt_col in rt_choices else 0
        rt_col_pick = st.selectbox("Expected RT column (optional)", rt_choices, index=default_rt_index)

    selected_library_column = None
    if not library_df.empty:
        st.caption("Optional: auto-fill expected RTs from your saved standard library.")
        library_columns = sorted(library_df["column_id"].dropna().unique().tolist())
        selected_library_column = st.selectbox(
            "Use expected RTs from saved library column",
            ["(none)"] + library_columns,
            index=0,
        )
        if selected_library_column != "(none)":
            metabolite_df = merge_expected_rts(
                metabolite_table=metabolite_df,
                column_id=selected_library_column,
                library=rt_library,
                metabolite_col=name_col,
            )
            rt_col_pick = "expected_rt_from_library_min"
            st.success(
                "Expected RTs were merged from your saved library for this column where metabolite names match."
            )

    metabolite_df[mz_col] = pd.to_numeric(metabolite_df[mz_col], errors="coerce")
    dropped = int(metabolite_df[mz_col].isna().sum())
    metabolite_df = metabolite_df[metabolite_df[mz_col].notna()].copy()
    if dropped:
        st.warning(f"Dropped {dropped} metabolite rows due to non-numeric m/z values.")

    if rt_col_pick != "(none)":
        metabolite_df[rt_col_pick] = pd.to_numeric(metabolite_df[rt_col_pick], errors="coerce")
        rt_col: str | None = rt_col_pick
    else:
        rt_col = None

    st.dataframe(metabolite_df.head(20), use_container_width=True)

    st.subheader("2) Upload mzXML standard runs")
    uploaded_runs = st.file_uploader(
        "mzXML files from one or more columns",
        type=["mzxml", "mzXML"],
        accept_multiple_files=True,
    )
    if not uploaded_runs:
        st.info("Upload one or more mzXML files to continue.")
        return

    default_mapping = pd.DataFrame(
        [
            {
                "file_name": file.name,
                "run_id": Path(file.name).stem,
                "column_id": infer_default_column(file.name),
            }
            for file in uploaded_runs
        ]
    )
    st.caption("Edit run and column labels before analysis.")
    mapping_df = st.data_editor(
        default_mapping,
        disabled=["file_name"],
        use_container_width=True,
        hide_index=True,
        key="run_mapping_editor",
    )

    for col in ("run_id", "column_id"):
        mapping_df[col] = mapping_df[col].astype(str).str.strip()
    mapping_df["run_id"] = mapping_df["run_id"].replace("", pd.NA).fillna(mapping_df["file_name"])
    mapping_df["column_id"] = mapping_df["column_id"].replace("", pd.NA).fillna(mapping_df["run_id"])

    current_signature = hash_inputs(
        files=uploaded_runs,
        metabolite_df=metabolite_df,
        mapping_df=mapping_df,
        tolerance=tolerance,
        tolerance_unit=tolerance_unit,
        rt_window_min=rt_window_min,
        name_col=name_col,
        mz_col=mz_col,
        rt_col=rt_col,
    )

    if "analysis_state" not in st.session_state:
        st.session_state["analysis_state"] = {}

    rerun_requested = st.button("Run column comparison", type="primary")
    if rerun_requested:
        with st.spinner("Running pyOpenMS extraction and peak analysis..."):
            try:
                metrics_df, chromatograms = run_analysis(
                    uploaded_runs=uploaded_runs,
                    mapping_df=mapping_df,
                    metabolite_df=metabolite_df,
                    name_col=name_col,
                    mz_col=mz_col,
                    rt_col=rt_col,
                    tolerance=tolerance,
                    tolerance_unit=tolerance_unit,
                    rt_window_min=rt_window_min,
                )
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                return

        st.session_state["analysis_state"] = {
            "signature": current_signature,
            "metrics_df": metrics_df,
            "chromatograms": chromatograms,
        }
        st.success("Analysis completed.")

    state = st.session_state.get("analysis_state", {})
    if not state or state.get("signature") != current_signature:
        st.info("Click **Run column comparison** to compute results with current inputs.")
        return

    metrics_df: pd.DataFrame = state["metrics_df"]
    chromatograms = state["chromatograms"]
    if metrics_df.empty:
        st.warning("No peak metrics were generated.")
        return

    st.subheader("3) Peak-shape comparison (side-by-side by column)")
    metabolite_options = sorted(metrics_df["metabolite"].unique().tolist())
    selected_metabolite = st.selectbox("Metabolite to visualize", metabolite_options, index=0)
    normalize_shape = st.checkbox(
        "Normalize each run to max=1 (shape-only view)",
        value=True,
        help="Useful when comparing peak shape independent of absolute signal.",
    )
    plot_df = build_plot_df(chromatograms, selected_metabolite)
    if not plot_df.empty:
        if normalize_shape:
            plot_df["intensity"] = (
                plot_df.groupby("run_id")["intensity"]
                .transform(lambda values: values / values.max() if values.max() > 0 else values)
            )
        fig = px.line(
            plot_df,
            x="time_min",
            y="intensity",
            color="run_id",
            facet_col="column_id",
            facet_col_wrap=2,
            title=f"Peak shape for {selected_metabolite}",
            labels={"time_min": "Retention time (min)", "intensity": "Intensity"},
        )
        fig.update_yaxes(matches=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No chromatogram points available for selected metabolite.")

    metabolite_metrics = (
        metrics_df[metrics_df["metabolite"] == selected_metabolite]
        .sort_values(["column_id", "run_id"])
        .reset_index(drop=True)
    )
    st.dataframe(
        metabolite_metrics[
            [
                "column_id",
                "run_id",
                "metabolite",
                "apex_rt_min",
                "fwhm_min",
                "asymmetry_10",
                "efficiency_plates",
                "snr",
                "peak_area",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("4) Column-level efficiency summary")
    summary_df = summarise_columns(metrics_df)
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("5) Recommend best column for target analytes")
    selected_targets = st.multiselect(
        "Metabolites the end user wants to measure",
        metabolite_options,
        default=metabolite_options[: min(5, len(metabolite_options))],
    )
    ranking_df = rank_columns(metrics_df, selected_targets)
    if ranking_df.empty:
        st.info("No ranking available for current selection.")
    else:
        st.dataframe(ranking_df, use_container_width=True)
        score_chart = px.bar(
            ranking_df,
            x="column_id",
            y="composite_score",
            title="Composite column score (current standard runs)",
        )
        st.plotly_chart(score_chart, use_container_width=True)

    st.subheader("6) Save retention times to standards library")
    st.caption(
        "This saves apex retention times from the current standards run into a local JSON library for future matching."
    )
    if st.button("Save current apex RTs to library"):
        updated_library = update_rt_library_from_metrics(
            library=rt_library,
            metrics=metrics_df,
            min_snr=min_snr_for_save,
        )
        save_rt_library(library_path, updated_library)
        st.success(f"Saved retention-time standards to {library_path}.")
        rt_library = updated_library
        library_df = rt_library_to_dataframe(rt_library)

    st.markdown("#### Saved RT library")
    if library_df.empty:
        st.info("No saved standards yet.")
    else:
        st.dataframe(
            library_df.sort_values(["column_id", "metabolite"]).reset_index(drop=True),
            use_container_width=True,
        )

        if selected_targets:
            st.markdown("#### Column fit using saved standards")
            fit_df = compute_library_fit(library_df, selected_targets=selected_targets)
            if fit_df.empty:
                st.info("Selected metabolites are not yet covered in the saved library.")
            else:
                st.dataframe(fit_df, use_container_width=True)
                st.plotly_chart(
                    px.bar(
                        fit_df,
                        x="column_id",
                        y="library_fit_score",
                        title="Column fit score from saved RT standards",
                    ),
                    use_container_width=True,
                )

    st.download_button(
        label="Download analysis metrics (CSV)",
        data=metrics_df.to_csv(index=False).encode(),
        file_name="column_comparison_metrics.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download retention library (JSON)",
        data=json.dumps(rt_library, indent=2).encode(),
        file_name="retention_library.json",
        mime="application/json",
    )


if __name__ == "__main__":
    app()
