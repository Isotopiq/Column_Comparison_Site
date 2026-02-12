from __future__ import annotations

import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from column_compare.analysis import (
    assess_peak_acceptability,
    build_acceptability_matrix,
    compute_peak_metrics,
    extract_chromatograms,
    load_ms1_experiment,
    parse_metabolite_table,
    preprocess_chromatogram,
    rank_columns,
    summarise_columns,
)
from column_compare.reports import (
    build_metabolite_report_html,
    build_metabolite_report_pdf,
    build_preview_image_bundle,
    zip_named_bytes,
)
from column_compare.storage import (
    load_rt_library,
    load_notes_store,
    merge_expected_rts,
    notes_store_to_dataframe,
    rt_library_to_dataframe,
    save_rt_library,
    save_notes_store,
    update_notes_store_from_dataframe,
    update_rt_library_from_metrics,
)

DEFAULT_LIBRARY_PATH = "data/retention_library.json"
DEFAULT_NOTES_PATH = "data/metabolite_notes.json"
DEFAULT_METABOLITE_TABLE_DIR = "data/metabolite_tables"


def infer_default_column(run_name: str) -> str:
    stem = Path(run_name).stem
    for delimiter in ("__", "_", "-", "."):
        if delimiter in stem:
            candidate = stem.split(delimiter)[0].strip()
            if candidate:
                return candidate
    return stem


def read_metabolite_table_bytes(raw_bytes: bytes, file_name: str) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw_bytes))

    try:
        return pd.read_csv(io.BytesIO(raw_bytes))
    except pd.errors.ParserError:
        return pd.read_csv(io.BytesIO(raw_bytes), sep=";")


def read_metabolite_table(uploaded_file) -> pd.DataFrame:
    return read_metabolite_table_bytes(uploaded_file.getvalue(), uploaded_file.name)


def read_metabolite_table_path(path: str | Path) -> pd.DataFrame:
    path_obj = Path(path)
    return read_metabolite_table_bytes(path_obj.read_bytes(), path_obj.name)


def _safe_filename_stem(name: str) -> str:
    clean = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in name)
    clean = clean.strip("_")
    return clean or "metabolites"


def persist_uploaded_metabolite_file(
    uploaded_file,
    target_directory: str = DEFAULT_METABOLITE_TABLE_DIR,
) -> Path:
    raw_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower() or ".csv"
    digest = hashlib.sha256(raw_bytes).hexdigest()[:12]
    stem = _safe_filename_stem(Path(uploaded_file.name).stem)
    target_dir = Path(target_directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{stem}_{digest}{suffix}"
    if not path.exists():
        path.write_bytes(raw_bytes)
    return path


def list_saved_metabolite_tables(
    target_directory: str = DEFAULT_METABOLITE_TABLE_DIR,
) -> list[Path]:
    target_dir = Path(target_directory)
    if not target_dir.exists():
        return []
    allowed = {".csv", ".txt", ".xlsx", ".xls"}
    paths = [p for p in target_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed]
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _guess_column(
    columns: list[str],
    preferred_exact: list[str],
    preferred_contains: list[str],
) -> str | None:
    normalized_map = {col.lower().strip(): col for col in columns}
    for candidate in preferred_exact:
        if candidate in normalized_map:
            return normalized_map[candidate]

    for col in columns:
        lowered = col.lower().strip()
        if any(token in lowered for token in preferred_contains):
            return col
    return None


def guess_metabolite_columns(columns: list[str]) -> tuple[str, str, str | None]:
    name_col = _guess_column(
        columns,
        preferred_exact=["metabolite", "compound", "analyte", "name"],
        preferred_contains=["metab", "compound", "analyte", "name"],
    )
    mz_col = _guess_column(
        columns,
        preferred_exact=["m/z", "mz", "mass_to_charge"],
        preferred_contains=["m/z", "mz"],
    )
    rt_col = _guess_column(
        columns,
        preferred_exact=["rt", "retention_time", "retention time"],
        preferred_contains=["retention", "rt"],
    )
    return (
        name_col or columns[0],
        mz_col or columns[0],
        rt_col,
    )


def coerce_numeric_series_with_fallback(values: pd.Series) -> pd.Series:
    as_text = values.astype(str).str.replace(",", ".", regex=False)
    coerced = pd.to_numeric(as_text, errors="coerce")
    if coerced.notna().all():
        return coerced

    extracted = as_text.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    fallback = pd.to_numeric(extracted, errors="coerce")
    return coerced.fillna(fallback)


def selected_rows_from_event(event: Any) -> list[int]:
    if event is None:
        return []
    try:
        rows = event.selection.rows
        return list(rows) if rows is not None else []
    except Exception:
        pass
    try:
        selection = event.get("selection", {})
        rows = selection.get("rows", [])
        return list(rows) if rows is not None else []
    except Exception:
        return []


def default_metabolite_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metabolite": ["Lactate", "Pyruvate", "Citrate", "Glucose"],
            "mz": [89.0244, 87.0088, 191.0197, 179.0556],
            "expected_rt_min": [2.3, 2.9, 4.1, 5.0],
        }
    )


def default_integration_bounds_table(
    metabolite_df: pd.DataFrame,
    metabolite_col: str,
    rt_col: str | None,
    default_half_window_min: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in metabolite_df.iterrows():
        metabolite = str(row[metabolite_col]).strip()
        expected_rt = None
        if rt_col is not None and rt_col in metabolite_df.columns:
            value = row[rt_col]
            if pd.notna(value):
                expected_rt = float(value)
        if expected_rt is None:
            start = None
            end = None
        else:
            start = expected_rt - float(default_half_window_min)
            end = expected_rt + float(default_half_window_min)
        rows.append(
            {
                "metabolite": metabolite,
                "integration_start_min": start,
                "integration_end_min": end,
            }
        )
    return pd.DataFrame(rows)


def integration_bounds_lookup(bounds_df: pd.DataFrame) -> dict[str, tuple[float | None, float | None]]:
    lookup: dict[str, tuple[float | None, float | None]] = {}
    if bounds_df.empty:
        return lookup

    prepared = bounds_df.copy()
    for col in ("integration_start_min", "integration_end_min"):
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    for _, row in prepared.iterrows():
        metabolite = str(row["metabolite"]).strip()
        if not metabolite:
            continue
        start = None if pd.isna(row["integration_start_min"]) else float(row["integration_start_min"])
        end = None if pd.isna(row["integration_end_min"]) else float(row["integration_end_min"])
        lookup[metabolite] = (start, end)
    return lookup


def hash_inputs(
    files: list[Any],
    metabolite_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    integration_bounds_df: pd.DataFrame,
    tolerance: float,
    tolerance_unit: str,
    rt_window_min: float,
    smoothing_window_points: int,
    baseline_mode: str,
    baseline_window_points: int,
    baseline_percentile: float,
    name_col: str,
    mz_col: str,
    rt_col: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(tolerance).encode())
    digest.update(tolerance_unit.encode())
    digest.update(str(rt_window_min).encode())
    digest.update(str(smoothing_window_points).encode())
    digest.update(baseline_mode.encode())
    digest.update(str(baseline_window_points).encode())
    digest.update(str(baseline_percentile).encode())
    digest.update(name_col.encode())
    digest.update(mz_col.encode())
    digest.update((rt_col or "").encode())
    digest.update(metabolite_df.to_csv(index=False).encode())
    digest.update(mapping_df.to_csv(index=False).encode())
    digest.update(integration_bounds_df.to_csv(index=False).encode())
    for uploaded in files:
        digest.update(uploaded.name.encode())
        digest.update(str(uploaded.size).encode())
        digest.update(uploaded.getvalue())
    return digest.hexdigest()


def build_plot_df(chromatograms: dict[tuple[str, str, str, str], Any], metabolite: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (run_id, column_id, file_name, metabolite_name), trace in chromatograms.items():
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
                "file_name": file_name,
            }
        )
        rows.append(frame)
    if not rows:
        return pd.DataFrame(
            columns=["time_min", "intensity", "run_id", "column_id", "file_name"]
        )
    return pd.concat(rows, ignore_index=True)


def make_peak_shape_figure(
    plot_df: pd.DataFrame,
    metabolite: str,
    normalize_shape: bool,
    facet_mode: str = "By column",
) -> tuple[Any | None, pd.DataFrame]:
    if plot_df.empty:
        return None, plot_df

    chart_df = plot_df.copy()
    if normalize_shape:
        chart_df["intensity"] = (
            chart_df.groupby("file_name")["intensity"]
            .transform(lambda values: values / values.max() if values.max() > 0 else values)
        )
    facet_col = "column_id" if facet_mode == "By column" else "file_name"
    fig = px.line(
        chart_df,
        x="time_min",
        y="intensity",
        color="file_name",
        facet_col=facet_col,
        facet_col_wrap=2,
        title=f"Peak shape for {metabolite}",
        labels={
            "time_min": "Retention time (min)",
            "intensity": "Intensity",
            "file_name": "mzXML file",
            "column_id": "Column",
        },
        hover_data={"run_id": True, "column_id": True, "file_name": True},
    )
    fig.update_yaxes(matches=None)
    return fig, chart_df


def build_table_preview_figure(title: str, df: pd.DataFrame, max_rows: int = 25) -> go.Figure:
    preview = df.head(max_rows).copy()
    preview = preview.fillna("")
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(preview.columns), fill_color="#E5E7EB", align="left"),
                cells=dict(
                    values=[preview[column].tolist() for column in preview.columns],
                    fill_color="#FFFFFF",
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    return fig


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
    integration_bounds: dict[str, tuple[float | None, float | None]],
    smoothing_window_points: int,
    baseline_mode: str,
    baseline_window_points: int,
    baseline_percentile: float,
    name_col: str,
    mz_col: str,
    rt_col: str | None,
    tolerance: float,
    tolerance_unit: str,
    rt_window_min: float,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str], Any], dict[tuple[str, str, str, str], Any]]:
    targets = parse_metabolite_table(metabolite_df, name_col=name_col, mz_col=mz_col, rt_col=rt_col)
    if not targets:
        raise ValueError("No valid metabolite rows were found after parsing your table.")

    uploaded_by_name = {file.name: file for file in uploaded_runs}
    metrics_rows: list[dict[str, Any]] = []
    raw_chromatograms: dict[tuple[str, str, str, str], Any] = {}
    processed_chromatograms: dict[tuple[str, str, str, str], Any] = {}

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
                raw_trace = trace_map[target.name]
                processed_trace = preprocess_chromatogram(
                    chromatogram=raw_trace,
                    smoothing_window_points=smoothing_window_points,
                    baseline_mode=baseline_mode,
                    baseline_window_points=baseline_window_points,
                    baseline_percentile=baseline_percentile,
                )
                bounds = integration_bounds.get(target.name, (None, None))
                metrics = compute_peak_metrics(
                    chromatogram=processed_trace,
                    run_id=run_id,
                    column_id=column_id,
                    target=target,
                    expected_rt_min=target.expected_rt_min,
                    rt_window_min=rt_window_min,
                    integration_start_min=bounds[0],
                    integration_end_min=bounds[1],
                )
                record = metrics.to_record()
                record["file_name"] = file_name
                metrics_rows.append(record)
                key = (run_id, column_id, file_name, target.name)
                raw_chromatograms[key] = raw_trace
                processed_chromatograms[key] = processed_trace
        finally:
            tmp_path.unlink(missing_ok=True)

    return pd.DataFrame(metrics_rows), raw_chromatograms, processed_chromatograms


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

        st.markdown("### Signal preprocessing")
        smoothing_enabled = st.checkbox(
            "Enable smoothing",
            value=True,
            help="Applies moving-average smoothing before peak picking.",
        )
        smoothing_window_points = int(
            st.number_input(
                "Smoothing window (points)",
                min_value=1,
                max_value=101,
                value=7,
                step=2,
            )
        )
        baseline_mode = st.selectbox(
            "Baseline correction",
            ["none", "rolling_min", "percentile"],
            index=1,
        )
        baseline_window_points = int(
            st.number_input(
                "Rolling baseline window (points)",
                min_value=3,
                max_value=401,
                value=31,
                step=2,
                disabled=baseline_mode != "rolling_min",
            )
        )
        baseline_percentile = float(
            st.number_input(
                "Percentile baseline (%)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                disabled=baseline_mode != "percentile",
            )
        )

        st.markdown("### Acceptable peak-shape thresholds")
        accept_min_snr = float(st.number_input("Min S/N", min_value=0.0, value=5.0, step=0.5))
        accept_max_fwhm = float(
            st.number_input("Max FWHM (min)", min_value=0.0, value=0.4, step=0.05)
        )
        col_left, col_right = st.columns(2)
        with col_left:
            accept_min_asym = float(
                st.number_input("Min asymmetry", min_value=0.0, value=0.7, step=0.1)
            )
        with col_right:
            accept_max_asym = float(
                st.number_input("Max asymmetry", min_value=0.0, value=1.8, step=0.1)
            )
        accept_min_efficiency = float(
            st.number_input(
                "Min theoretical plates",
                min_value=0.0,
                value=1000.0,
                step=100.0,
            )
        )

        st.markdown("### Persistence")
        min_snr_for_save = st.number_input(
            "Minimum S/N to save RT standards",
            min_value=0.0,
            value=3.0,
            step=0.5,
        )
        library_path = st.text_input("Retention library file", value=DEFAULT_LIBRARY_PATH)
        notes_path = st.text_input("Metabolite notes file", value=DEFAULT_NOTES_PATH)

    smoothing_window_points = smoothing_window_points if smoothing_enabled else 1
    if accept_min_asym > accept_max_asym:
        accept_min_asym, accept_max_asym = accept_max_asym, accept_min_asym
        st.sidebar.warning("Asymmetry bounds were swapped to keep min <= max.")

    rt_library = load_rt_library(library_path)
    library_df = rt_library_to_dataframe(rt_library)
    notes_store = load_notes_store(notes_path)
    if "metabolite_setup" not in st.session_state:
        st.session_state["metabolite_setup"] = None

    st.subheader("1) Upload or select metabolite list")
    st.caption(
        "Supported formats include columns like `compound`, `m/z`, and `RT`. "
        "Click **Confirm metabolite table and continue** to unlock mzXML upload."
    )

    source_mode = st.radio(
        "Metabolite source",
        ["Upload new file", "Use saved file"],
        horizontal=True,
        key="metabolite_source_mode",
    )

    candidate_metabolite_df: pd.DataFrame | None = None
    candidate_source_name: str | None = None
    saved_tables = list_saved_metabolite_tables()

    if source_mode == "Upload new file":
        metabolite_file = st.file_uploader(
            "Metabolite table (CSV/Excel)",
            type=["csv", "txt", "xlsx", "xls"],
            accept_multiple_files=False,
        )
        if metabolite_file is not None:
            try:
                candidate_metabolite_df = read_metabolite_table(metabolite_file)
            except Exception as exc:
                st.error(f"Failed to read metabolite table: {exc}")
            else:
                cached_path = persist_uploaded_metabolite_file(metabolite_file)
                candidate_source_name = metabolite_file.name
                st.caption(f"Saved for reuse as: `{cached_path.name}`")
    else:
        if not saved_tables:
            st.info("No saved metabolite tables found yet. Upload one to create reusable history.")
        else:
            saved_names = [path.name for path in saved_tables]
            selected_saved_name = st.selectbox(
                "Previously uploaded metabolite tables",
                saved_names,
                index=0,
            )
            selected_saved_path = next(path for path in saved_tables if path.name == selected_saved_name)
            try:
                candidate_metabolite_df = read_metabolite_table_path(selected_saved_path)
            except Exception as exc:
                st.error(f"Failed to read saved metabolite table: {exc}")
            else:
                candidate_source_name = selected_saved_name
                st.caption(f"Loaded from `{selected_saved_path}`")

    if candidate_metabolite_df is not None:
        if candidate_metabolite_df.empty:
            st.warning("Selected metabolite table is empty.")
        else:
            metabolite_df = candidate_metabolite_df.copy()
            all_columns = list(metabolite_df.columns)
            guessed_name_col, guessed_mz_col, guessed_rt_col = guess_metabolite_columns(all_columns)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                name_col = st.selectbox(
                    "Metabolite name column",
                    all_columns,
                    index=all_columns.index(guessed_name_col),
                    key="metabolite_name_column_select",
                )
            with col_b:
                mz_col = st.selectbox(
                    "m/z column",
                    all_columns,
                    index=all_columns.index(guessed_mz_col),
                    key="metabolite_mz_column_select",
                )
            with col_c:
                rt_choices = ["(none)"] + all_columns
                default_rt_index = rt_choices.index(guessed_rt_col) if guessed_rt_col in rt_choices else 0
                rt_col_pick = st.selectbox(
                    "Expected RT column (optional)",
                    rt_choices,
                    index=default_rt_index,
                    key="metabolite_rt_column_select",
                )

            if not library_df.empty:
                st.caption("Optional: auto-fill expected RTs from your saved standard library.")
                library_columns = sorted(library_df["column_id"].dropna().unique().tolist())
                selected_library_column = st.selectbox(
                    "Use expected RTs from saved library column",
                    ["(none)"] + library_columns,
                    index=0,
                    key="metabolite_library_column_select",
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

            metabolite_df[mz_col] = coerce_numeric_series_with_fallback(metabolite_df[mz_col])
            dropped = int(metabolite_df[mz_col].isna().sum())
            metabolite_df = metabolite_df[metabolite_df[mz_col].notna()].copy()
            if dropped:
                st.warning(f"Dropped {dropped} metabolite rows due to non-numeric m/z values.")

            if rt_col_pick != "(none)":
                metabolite_df[rt_col_pick] = coerce_numeric_series_with_fallback(metabolite_df[rt_col_pick])
                invalid_rt = int(metabolite_df[rt_col_pick].isna().sum())
                rt_col: str | None = rt_col_pick
                if invalid_rt:
                    st.caption(
                        f"{invalid_rt} rows have missing or non-numeric RT values and will be treated as unknown."
                    )
            else:
                rt_col = None

            st.dataframe(metabolite_df.head(20), use_container_width=True)
            if st.button("Confirm metabolite table and continue", type="primary"):
                st.session_state["metabolite_setup"] = {
                    "source_name": candidate_source_name or "uploaded_table",
                    "name_col": name_col,
                    "mz_col": mz_col,
                    "rt_col": rt_col,
                    "metabolite_df_json": metabolite_df.to_json(orient="split"),
                }
                st.session_state["analysis_state"] = {}
                st.success("Metabolite table confirmed. Continue with mzXML upload below.")

    active_metabolite_setup = st.session_state.get("metabolite_setup")
    if active_metabolite_setup is None:
        st.info("Confirm a metabolite table to unlock mzXML upload and viewer.")
        template_df = default_metabolite_template()
        st.dataframe(template_df, use_container_width=True)
        st.download_button(
            label="Download metabolite template (CSV)",
            data=template_df.to_csv(index=False).encode(),
            file_name="metabolite_template.csv",
            mime="text/csv",
        )
        return

    metabolite_df = pd.read_json(io.StringIO(active_metabolite_setup["metabolite_df_json"]), orient="split")
    name_col = str(active_metabolite_setup["name_col"])
    mz_col = str(active_metabolite_setup["mz_col"])
    rt_col = active_metabolite_setup["rt_col"]
    st.success(
        f"Active metabolite table: `{active_metabolite_setup['source_name']}` "
        f"({len(metabolite_df)} metabolites with valid m/z)."
    )
    if st.button("Clear active metabolite table"):
        st.session_state["metabolite_setup"] = None
        st.session_state["analysis_state"] = {}
        st.rerun()

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
    st.caption(
        "Assign each mzXML file to the correct column. "
        "Each run_id must be unique so chromatograms are tracked accurately."
    )
    default_column_options = sorted(default_mapping["column_id"].dropna().astype(str).unique().tolist())
    custom_column_text = st.text_input(
        "Column IDs (comma-separated, used as assignment options)",
        value=", ".join(default_column_options),
        help="Example: HILIC_A, HILIC_B, RP_18",
    )
    custom_column_options = [
        token.strip() for token in custom_column_text.split(",") if token.strip()
    ]
    column_options = sorted(set(default_column_options + custom_column_options))
    if not column_options:
        column_options = ["Column_1"]

    mapping_df = st.data_editor(
        default_mapping,
        disabled=["file_name"],
        column_config={
            "column_id": st.column_config.SelectboxColumn(
                "column_id",
                options=column_options if column_options else default_column_options,
                required=True,
            ),
        },
        use_container_width=True,
        hide_index=True,
        key="run_mapping_editor",
    )

    for col in ("run_id", "column_id"):
        mapping_df[col] = mapping_df[col].astype(str).str.strip()
    mapping_df["run_id"] = mapping_df["run_id"].replace("", pd.NA).fillna(mapping_df["file_name"])
    mapping_df["column_id"] = mapping_df["column_id"].replace("", pd.NA).fillna(mapping_df["run_id"])

    duplicate_runs = mapping_df[mapping_df["run_id"].duplicated(keep=False)]["run_id"].tolist()
    duplicate_runs = sorted(set(duplicate_runs))
    if duplicate_runs:
        st.error(
            "Duplicate run_id values found. Please make run_id unique per mzXML file before running analysis."
        )
        st.dataframe(
            mapping_df[mapping_df["run_id"].isin(duplicate_runs)][
                ["file_name", "run_id", "column_id"]
            ],
            use_container_width=True,
        )
        return

    if mapping_df["column_id"].astype(str).str.strip().eq("").any():
        st.error("Each mzXML file must be assigned to a column_id.")
        return

    mapping_summary = (
        mapping_df.groupby("column_id", dropna=False)
        .agg(
            n_files=("file_name", "count"),
            files=("file_name", lambda values: ", ".join(sorted(values))),
        )
        .reset_index()
        .sort_values("column_id")
    )
    st.markdown("**Current file-to-column assignment**")
    st.dataframe(mapping_summary, use_container_width=True)

    st.subheader("3) Edit peak integration bounds")
    st.caption(
        "Optional manual integration windows by metabolite. Leave blank for full chromatogram."
    )
    default_bounds = default_integration_bounds_table(
        metabolite_df=metabolite_df,
        metabolite_col=name_col,
        rt_col=rt_col,
        default_half_window_min=rt_window_min if rt_window_min > 0 else 0.5,
    )
    bounds_df = st.data_editor(
        default_bounds,
        disabled=["metabolite"],
        use_container_width=True,
        hide_index=True,
        key="integration_bounds_editor",
    )
    integration_bounds = integration_bounds_lookup(bounds_df)

    current_signature = hash_inputs(
        files=uploaded_runs,
        metabolite_df=metabolite_df,
        mapping_df=mapping_df,
        integration_bounds_df=bounds_df,
        tolerance=tolerance,
        tolerance_unit=tolerance_unit,
        rt_window_min=rt_window_min,
        smoothing_window_points=smoothing_window_points,
        baseline_mode=baseline_mode,
        baseline_window_points=baseline_window_points,
        baseline_percentile=baseline_percentile,
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
                metrics_df, raw_chromatograms, processed_chromatograms = run_analysis(
                    uploaded_runs=uploaded_runs,
                    mapping_df=mapping_df,
                    metabolite_df=metabolite_df,
                    integration_bounds=integration_bounds,
                    smoothing_window_points=smoothing_window_points,
                    baseline_mode=baseline_mode,
                    baseline_window_points=baseline_window_points,
                    baseline_percentile=baseline_percentile,
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
            "raw_chromatograms": raw_chromatograms,
            "processed_chromatograms": processed_chromatograms,
        }
        st.success("Analysis completed.")

    state = st.session_state.get("analysis_state", {})
    if not state or state.get("signature") != current_signature:
        st.info("Click **Run column comparison** to compute results with current inputs.")
        return

    metrics_df: pd.DataFrame = state["metrics_df"]
    raw_chromatograms = state["raw_chromatograms"]
    processed_chromatograms = state["processed_chromatograms"]
    if metrics_df.empty:
        st.warning("No peak metrics were generated.")
        return

    scored_metrics_df = assess_peak_acceptability(
        metrics=metrics_df,
        min_snr=accept_min_snr,
        max_fwhm_min=accept_max_fwhm,
        min_asymmetry_10=accept_min_asym,
        max_asymmetry_10=accept_max_asym,
        min_efficiency_plates=accept_min_efficiency,
    )

    metabolite_options = sorted(scored_metrics_df["metabolite"].unique().tolist())
    notes_map = {
        metabolite: payload.get("note", "")
        for metabolite, payload in notes_store.get("notes", {}).items()
    }

    st.subheader("4) Peak-shape comparison (side-by-side by column)")
    metabolite_browser_df = (
        scored_metrics_df.groupby("metabolite", dropna=False)
        .agg(
            n_runs=("run_id", "nunique"),
            n_columns=("column_id", "nunique"),
            detected_runs=("has_peak", "sum"),
            acceptable_runs=("is_acceptable_shape", "sum"),
            median_apex_rt_min=("apex_rt_min", "median"),
        )
        .reset_index()
        .sort_values("metabolite")
        .reset_index(drop=True)
    )
    metabolite_browser_df["acceptable_pct"] = (
        metabolite_browser_df["acceptable_runs"] / metabolite_browser_df["n_runs"].clip(lower=1) * 100.0
    )
    st.caption(
        "Click a metabolite row below to update the chromatogram view. "
        "This shows traces from each mapped column file."
    )
    metabolite_event = st.dataframe(
        metabolite_browser_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="metabolite_click_table",
    )
    if (
        "selected_metabolite_name" not in st.session_state
        or st.session_state["selected_metabolite_name"] not in metabolite_options
    ):
        st.session_state["selected_metabolite_name"] = metabolite_options[0]

    clicked_rows = selected_rows_from_event(metabolite_event)
    if clicked_rows:
        row_idx = int(clicked_rows[0])
        if 0 <= row_idx < len(metabolite_browser_df):
            st.session_state["selected_metabolite_name"] = str(
                metabolite_browser_df.iloc[row_idx]["metabolite"]
            )

    selected_metabolite = st.selectbox(
        "Selected metabolite",
        metabolite_options,
        key="selected_metabolite_name",
    )

    facet_mode = st.radio(
        "Chromatogram layout",
        ["By column", "By file"],
        horizontal=True,
        help="By column groups traces by assigned column_id. By file shows one panel per mzXML file.",
    )
    trace_view_mode = st.radio(
        "Trace view",
        ["Processed trace", "Raw trace"],
        index=0,
        horizontal=True,
    )
    normalize_shape = st.checkbox(
        "Normalize each run to max=1 (shape-only view)",
        value=True,
        help="Useful when comparing peak shape independent of absolute signal.",
    )
    active_chromatograms = (
        processed_chromatograms if trace_view_mode == "Processed trace" else raw_chromatograms
    )
    plot_df = build_plot_df(active_chromatograms, selected_metabolite)
    peak_shape_fig = None
    if not plot_df.empty:
        peak_shape_fig, _ = make_peak_shape_figure(
            plot_df=plot_df,
            metabolite=selected_metabolite,
            normalize_shape=normalize_shape,
            facet_mode=facet_mode,
        )
        bounds = integration_bounds.get(selected_metabolite, (None, None))
        if peak_shape_fig is not None:
            if bounds[0] is not None:
                peak_shape_fig.add_vline(
                    x=bounds[0],
                    line_dash="dash",
                    line_color="#6B7280",
                    annotation_text="int start",
                )
            if bounds[1] is not None:
                peak_shape_fig.add_vline(
                    x=bounds[1],
                    line_dash="dash",
                    line_color="#6B7280",
                    annotation_text="int end",
                )
            st.plotly_chart(peak_shape_fig, use_container_width=True)
    else:
        st.warning("No chromatogram points available for selected metabolite.")

    metabolite_metrics = (
        scored_metrics_df[scored_metrics_df["metabolite"] == selected_metabolite]
        .sort_values(["column_id", "run_id"])
        .reset_index(drop=True)
    )
    metabolite_metrics["note"] = metabolite_metrics["metabolite"].map(notes_map).fillna("")
    if notes_map.get(selected_metabolite):
        st.info(f"Saved note for {selected_metabolite}: {notes_map[selected_metabolite]}")
    st.dataframe(
        metabolite_metrics[
            [
                "file_name",
                "column_id",
                "run_id",
                "metabolite",
                "apex_rt_min",
                "fwhm_min",
                "asymmetry_10",
                "efficiency_plates",
                "snr",
                "peak_area",
                "is_acceptable_shape",
                "acceptability_reason",
                "integration_start_min",
                "integration_end_min",
                "note",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("5) Column-level efficiency summary")
    summary_df = summarise_columns(scored_metrics_df)
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("6) Recommend best column for target analytes")
    selected_targets = st.multiselect(
        "Metabolites the end user wants to measure",
        metabolite_options,
        default=metabolite_options[: min(5, len(metabolite_options))],
    )
    ranking_df = rank_columns(scored_metrics_df, selected_targets)
    ranking_fig = None
    if ranking_df.empty:
        st.info("No ranking available for current selection.")
    else:
        st.dataframe(ranking_df, use_container_width=True)
        ranking_fig = px.bar(
            ranking_df,
            x="column_id",
            y="composite_score",
            title="Composite column score (current standard runs)",
        )
        st.plotly_chart(ranking_fig, use_container_width=True)

    st.subheader("7) Heatmap: acceptable peak shape by metabolite and column")
    st.caption(
        "Cell values represent % of runs in each column where the metabolite passes current acceptability thresholds."
    )
    heatmap_df = build_acceptability_matrix(scored_metrics_df)
    heatmap_fig = None
    if heatmap_df.empty:
        st.info("No heatmap available yet. Check that runs and metabolites were analyzed.")
    else:
        heatmap_fig = px.imshow(
            heatmap_df,
            text_auto=".0f",
            color_continuous_scale=["#DC2626", "#F59E0B", "#16A34A"],
            range_color=[0, 100],
            labels={"x": "Column", "y": "Metabolite", "color": "Acceptable (%)"},
            title="Peak-shape acceptability heatmap",
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.dataframe(heatmap_df, use_container_width=True)

    st.subheader("8) Metabolite notes")
    st.caption("Add notes for each metabolite. Notes are saved and available in future sessions.")
    notes_editor_df = notes_store_to_dataframe(notes_store, metabolites=metabolite_options)
    notes_editor_df = notes_editor_df[["metabolite", "note", "last_updated"]]
    edited_notes_df = st.data_editor(
        notes_editor_df,
        disabled=["metabolite", "last_updated"],
        hide_index=True,
        use_container_width=True,
        key="metabolite_notes_editor",
    )
    if st.button("Save metabolite notes"):
        updated_notes = update_notes_store_from_dataframe(notes_store, edited_notes_df)
        save_notes_store(notes_path, updated_notes)
        st.success(f"Saved metabolite notes to {notes_path}.")
        notes_store = updated_notes
        notes_map = {
            metabolite: payload.get("note", "")
            for metabolite, payload in notes_store.get("notes", {}).items()
        }

    st.subheader("9) Save retention times to standards library")
    st.caption(
        "This saves apex retention times from the current standards run into a local JSON library for future matching."
    )
    if st.button("Save current apex RTs to library"):
        updated_library = update_rt_library_from_metrics(
            library=rt_library,
            metrics=scored_metrics_df,
            min_snr=min_snr_for_save,
        )
        save_rt_library(library_path, updated_library)
        st.success(f"Saved retention-time standards to {library_path}.")
        rt_library = updated_library
        library_df = rt_library_to_dataframe(rt_library)

    library_fit_fig = None
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
                library_fit_fig = px.bar(
                    fit_df,
                    x="column_id",
                    y="library_fit_score",
                    title="Column fit score from saved RT standards",
                )
                st.plotly_chart(library_fit_fig, use_container_width=True)

    st.subheader("10) Export per-metabolite comparison report (HTML/PDF)")
    report_metabolite = st.selectbox(
        "Metabolite for export",
        metabolite_options,
        index=metabolite_options.index(selected_metabolite)
        if selected_metabolite in metabolite_options
        else 0,
        key="report_metabolite_select",
    )
    if st.button("Generate report files"):
        report_metrics = (
            scored_metrics_df[scored_metrics_df["metabolite"] == report_metabolite]
            .sort_values(["column_id", "run_id"])
            .reset_index(drop=True)
        )
        report_plot_df = build_plot_df(processed_chromatograms, report_metabolite)
        report_fig, _ = make_peak_shape_figure(
            plot_df=report_plot_df,
            metabolite=report_metabolite,
            normalize_shape=False,
        )
        if report_fig is None:
            st.error("Cannot generate report figure for selected metabolite.")
        else:
            report_settings = {
                "m/z tolerance": f"{tolerance} {tolerance_unit}",
                "expected RT window (+/- min)": rt_window_min,
                "smoothing window (points)": smoothing_window_points,
                "baseline mode": baseline_mode,
                "baseline window (points)": baseline_window_points,
                "baseline percentile": baseline_percentile,
                "accept min S/N": accept_min_snr,
                "accept max FWHM (min)": accept_max_fwhm,
                "accept asymmetry range": f"{accept_min_asym} - {accept_max_asym}",
                "accept min plates": accept_min_efficiency,
            }
            note = notes_map.get(report_metabolite, "")
            html_bytes = build_metabolite_report_html(
                metabolite=report_metabolite,
                metabolite_metrics=report_metrics,
                peak_shape_figure=report_fig,
                note=note,
                settings=report_settings,
            )
            pdf_bytes: bytes | None = None
            try:
                pdf_bytes = build_metabolite_report_pdf(
                    metabolite=report_metabolite,
                    metabolite_metrics=report_metrics,
                    peak_shape_figure=report_fig,
                    note=note,
                    settings=report_settings,
                )
            except RuntimeError as exc:
                st.warning(str(exc))

            st.session_state["report_bundle"] = {
                "metabolite": report_metabolite,
                "html_bytes": html_bytes,
                "pdf_bytes": pdf_bytes,
            }
            st.success("Report files generated.")

    report_bundle = st.session_state.get("report_bundle", {})
    if report_bundle:
        metabolite_label = report_bundle.get("metabolite", "metabolite")
        st.download_button(
            label="Download metabolite report (HTML)",
            data=report_bundle["html_bytes"],
            file_name=f"{metabolite_label}_comparison_report.html",
            mime="text/html",
        )
        if report_bundle.get("pdf_bytes") is not None:
            st.download_button(
                label="Download metabolite report (PDF)",
                data=report_bundle["pdf_bytes"],
                file_name=f"{metabolite_label}_comparison_report.pdf",
                mime="application/pdf",
            )

    st.subheader("11) Generate feature preview images")
    st.caption(
        "Creates preview PNGs for major app features and packages them in a ZIP file."
    )
    if st.button("Generate preview image pack"):
        preview_figures: dict[str, Any] = {}
        if peak_shape_fig is not None:
            preview_figures["feature_peak_shape_comparison"] = peak_shape_fig
        if ranking_fig is not None:
            preview_figures["feature_column_ranking"] = ranking_fig
        if heatmap_fig is not None:
            preview_figures["feature_acceptability_heatmap"] = heatmap_fig
        if library_fit_fig is not None:
            preview_figures["feature_library_fit"] = library_fit_fig

        bounds_preview_fig = build_table_preview_figure(
            title="Integration bounds preview",
            df=bounds_df[["metabolite", "integration_start_min", "integration_end_min"]],
        )
        preview_figures["feature_integration_bounds"] = bounds_preview_fig

        notes_preview_fig = build_table_preview_figure(
            title="Metabolite notes preview",
            df=edited_notes_df[["metabolite", "note", "last_updated"]],
        )
        preview_figures["feature_metabolite_notes"] = notes_preview_fig

        images = build_preview_image_bundle(preview_figures)
        if not images:
            st.warning(
                "Could not generate preview PNG images in this environment. "
                "Ensure plotly image export is available (kaleido)."
            )
        else:
            preview_zip = zip_named_bytes(images)
            st.download_button(
                label="Download preview images (ZIP)",
                data=preview_zip,
                file_name="feature_previews.zip",
                mime="application/zip",
            )
            for image_name, image_bytes in images.items():
                st.image(image_bytes, caption=image_name, use_container_width=True)

    st.download_button(
        label="Download analysis metrics (CSV)",
        data=scored_metrics_df.to_csv(index=False).encode(),
        file_name="column_comparison_metrics.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download retention library (JSON)",
        data=json.dumps(rt_library, indent=2).encode(),
        file_name="retention_library.json",
        mime="application/json",
    )
    st.download_button(
        label="Download metabolite notes (JSON)",
        data=json.dumps(notes_store, indent=2).encode(),
        file_name="metabolite_notes.json",
        mime="application/json",
    )


if __name__ == "__main__":
    app()
