from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MetaboliteTarget:
    name: str
    mz: float
    expected_rt_min: float | None = None


@dataclass(slots=True)
class PeakMetrics:
    run_id: str
    column_id: str
    metabolite: str
    mz: float
    apex_rt_min: float | None
    apex_intensity: float
    peak_area: float
    fwhm_min: float | None
    asymmetry_10: float | None
    efficiency_plates: float | None
    snr: float | None
    has_peak: bool
    integration_start_min: float | None = None
    integration_end_min: float | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Chromatogram:
    time_min: np.ndarray
    intensity: np.ndarray


def load_ms1_experiment(file_path: str | Path):
    """Load an mzXML file into an OpenMS experiment."""
    try:
        import pyopenms as oms
    except ImportError as exc:  # pragma: no cover - depends on runtime install
        raise RuntimeError(
            "pyOpenMS is required but is not installed. Install dependencies from requirements.txt."
        ) from exc

    exp = oms.MSExperiment()
    oms.MzXMLFile().load(str(file_path), exp)
    return exp


def parse_metabolite_table(
    table: pd.DataFrame, name_col: str, mz_col: str, rt_col: str | None = None
) -> list[MetaboliteTarget]:
    targets: list[MetaboliteTarget] = []
    for _, row in table.iterrows():
        name = str(row[name_col]).strip()
        if not name:
            continue

        mz = float(row[mz_col])
        expected_rt = None
        if rt_col is not None and rt_col in table.columns:
            value = row[rt_col]
            if pd.notna(value):
                expected_rt = float(value)

        targets.append(MetaboliteTarget(name=name, mz=mz, expected_rt_min=expected_rt))
    return targets


def extract_chromatograms(
    experiment,
    targets: list[MetaboliteTarget],
    tolerance: float,
    tolerance_unit: str = "ppm",
) -> dict[str, Chromatogram]:
    """Extract EIC traces for each metabolite target."""
    if tolerance <= 0:
        raise ValueError("m/z tolerance must be greater than zero.")

    times: list[float] = []
    trace_map = {target.name: [] for target in targets}

    for spectrum in experiment:
        if spectrum.getMSLevel() != 1:
            continue

        mzs, intensities = spectrum.get_peaks()
        if mzs.size == 0:
            continue

        times.append(float(spectrum.getRT()) / 60.0)
        for target in targets:
            if tolerance_unit == "ppm":
                mz_delta = target.mz * tolerance / 1_000_000.0
            else:
                mz_delta = tolerance
            mask = (mzs >= target.mz - mz_delta) & (mzs <= target.mz + mz_delta)
            value = float(np.sum(intensities[mask])) if np.any(mask) else 0.0
            trace_map[target.name].append(value)

    chromatograms: dict[str, Chromatogram] = {}
    time_array = np.asarray(times, dtype=float)
    for target in targets:
        chromatograms[target.name] = Chromatogram(
            time_min=time_array,
            intensity=np.asarray(trace_map[target.name], dtype=float),
        )
    return chromatograms


def _coerce_window(window_points: int, min_points: int = 1) -> int:
    window = max(int(window_points), min_points)
    if window % 2 == 0:
        window += 1
    return window


def _moving_average(values: np.ndarray, window_points: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    window = _coerce_window(window_points, min_points=1)
    if window <= 1:
        return values.copy()

    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def preprocess_chromatogram(
    chromatogram: Chromatogram,
    smoothing_window_points: int = 1,
    baseline_mode: Literal["none", "rolling_min", "percentile"] = "none",
    baseline_window_points: int = 21,
    baseline_percentile: float = 10.0,
) -> Chromatogram:
    """Apply optional smoothing and baseline correction."""
    y = np.asarray(chromatogram.intensity, dtype=float)
    if y.size == 0:
        return Chromatogram(
            time_min=np.asarray(chromatogram.time_min, dtype=float),
            intensity=y,
        )

    smoothed = _moving_average(y, smoothing_window_points)

    if baseline_mode == "rolling_min":
        baseline_window = _coerce_window(baseline_window_points, min_points=3)
        baseline = (
            pd.Series(smoothed)
            .rolling(window=baseline_window, center=True, min_periods=1)
            .min()
            .to_numpy(dtype=float)
        )
    elif baseline_mode == "percentile":
        percentile = float(np.clip(baseline_percentile, 0.0, 100.0))
        baseline = np.full_like(smoothed, np.percentile(smoothed, percentile))
    else:
        baseline = np.zeros_like(smoothed)

    corrected = np.clip(smoothed - baseline, a_min=0.0, a_max=None)
    return Chromatogram(
        time_min=np.asarray(chromatogram.time_min, dtype=float),
        intensity=corrected,
    )


def _linear_interpolate_x(
    x1: float, y1: float, x2: float, y2: float, y_target: float
) -> float | None:
    if y1 == y2:
        return None
    ratio = (y_target - y1) / (y2 - y1)
    if ratio < 0 or ratio > 1:
        return None
    return x1 + ratio * (x2 - x1)


def _find_crossing_left(
    x: np.ndarray, y: np.ndarray, apex_idx: int, y_target: float
) -> float | None:
    for idx in range(apex_idx, 0, -1):
        y_left = y[idx - 1]
        y_right = y[idx]
        if (y_left <= y_target <= y_right) or (y_left >= y_target >= y_right):
            crossing = _linear_interpolate_x(
                x[idx - 1], y_left, x[idx], y_right, y_target
            )
            if crossing is not None:
                return crossing
    return None


def _find_crossing_right(
    x: np.ndarray, y: np.ndarray, apex_idx: int, y_target: float
) -> float | None:
    for idx in range(apex_idx, len(x) - 1):
        y_left = y[idx]
        y_right = y[idx + 1]
        if (y_left >= y_target >= y_right) or (y_left <= y_target <= y_right):
            crossing = _linear_interpolate_x(
                x[idx], y_left, x[idx + 1], y_right, y_target
            )
            if crossing is not None:
                return crossing
    return None


def _estimate_noise(y: np.ndarray) -> float | None:
    if y.size < 8:
        return None
    baseline_mask = y <= np.percentile(y, 35)
    baseline_values = y[baseline_mask]
    if baseline_values.size < 4:
        return None
    noise = float(np.std(baseline_values))
    return noise if noise > 0 else None


def compute_peak_metrics(
    chromatogram: Chromatogram,
    run_id: str,
    column_id: str,
    target: MetaboliteTarget,
    expected_rt_min: float | None = None,
    rt_window_min: float | None = None,
    integration_start_min: float | None = None,
    integration_end_min: float | None = None,
) -> PeakMetrics:
    """Estimate peak-shape and efficiency metrics from an extracted chromatogram."""
    x = chromatogram.time_min
    y = chromatogram.intensity

    left_bound = integration_start_min
    right_bound = integration_end_min
    if (
        left_bound is not None
        and right_bound is not None
        and float(left_bound) > float(right_bound)
    ):
        left_bound, right_bound = right_bound, left_bound

    if x.size == 0 or y.size == 0:
        return PeakMetrics(
            run_id=run_id,
            column_id=column_id,
            metabolite=target.name,
            mz=target.mz,
            apex_rt_min=None,
            apex_intensity=0.0,
            peak_area=0.0,
            fwhm_min=None,
            asymmetry_10=None,
            efficiency_plates=None,
            snr=None,
            has_peak=False,
            integration_start_min=left_bound,
            integration_end_min=right_bound,
        )

    integration_mask = np.ones_like(x, dtype=bool)
    if left_bound is not None:
        integration_mask &= x >= float(left_bound)
    if right_bound is not None:
        integration_mask &= x <= float(right_bound)

    if not np.any(integration_mask):
        return PeakMetrics(
            run_id=run_id,
            column_id=column_id,
            metabolite=target.name,
            mz=target.mz,
            apex_rt_min=None,
            apex_intensity=0.0,
            peak_area=0.0,
            fwhm_min=None,
            asymmetry_10=None,
            efficiency_plates=None,
            snr=None,
            has_peak=False,
            integration_start_min=left_bound,
            integration_end_min=right_bound,
        )

    x_segment = x[integration_mask]
    y_segment = y[integration_mask]

    if np.all(y_segment <= 0):
        return PeakMetrics(
            run_id=run_id,
            column_id=column_id,
            metabolite=target.name,
            mz=target.mz,
            apex_rt_min=None,
            apex_intensity=0.0,
            peak_area=0.0,
            fwhm_min=None,
            asymmetry_10=None,
            efficiency_plates=None,
            snr=None,
            has_peak=False,
            integration_start_min=left_bound,
            integration_end_min=right_bound,
        )

    search_mask = np.ones_like(x_segment, dtype=bool)
    if expected_rt_min is not None and rt_window_min is not None and rt_window_min > 0:
        search_mask = np.abs(x_segment - expected_rt_min) <= rt_window_min
        if not np.any(search_mask):
            search_mask = np.ones_like(x_segment, dtype=bool)

    masked_indices = np.where(search_mask)[0]
    local_values = y_segment[masked_indices]
    apex_local_idx = int(np.argmax(local_values))
    apex_idx = int(masked_indices[apex_local_idx])

    apex_rt = float(x_segment[apex_idx])
    apex_intensity = float(y_segment[apex_idx])

    baseline = float(np.percentile(y_segment, 10))
    area = float(np.trapz(np.clip(y_segment - baseline, a_min=0, a_max=None), x_segment))

    half_height = baseline + (apex_intensity - baseline) * 0.5
    left_hh = _find_crossing_left(x_segment, y_segment, apex_idx, half_height)
    right_hh = _find_crossing_right(x_segment, y_segment, apex_idx, half_height)
    fwhm = None
    if left_hh is not None and right_hh is not None and right_hh > left_hh:
        fwhm = float(right_hh - left_hh)

    ten_pct_height = baseline + (apex_intensity - baseline) * 0.1
    left_10 = _find_crossing_left(x_segment, y_segment, apex_idx, ten_pct_height)
    right_10 = _find_crossing_right(x_segment, y_segment, apex_idx, ten_pct_height)
    asymmetry = None
    if (
        left_10 is not None
        and right_10 is not None
        and apex_rt > left_10
        and right_10 > apex_rt
    ):
        front = apex_rt - left_10
        back = right_10 - apex_rt
        if front > 0:
            asymmetry = float(back / front)

    plates = None
    if fwhm is not None and fwhm > 0:
        plates = float(5.54 * (apex_rt / fwhm) ** 2)

    noise = _estimate_noise(y_segment)
    snr = None if noise is None else float((apex_intensity - baseline) / noise)

    has_peak = bool(apex_intensity > baseline and area > 0)

    return PeakMetrics(
        run_id=run_id,
        column_id=column_id,
        metabolite=target.name,
        mz=target.mz,
        apex_rt_min=apex_rt,
        apex_intensity=apex_intensity,
        peak_area=area,
        fwhm_min=fwhm,
        asymmetry_10=asymmetry,
        efficiency_plates=plates,
        snr=snr,
        has_peak=has_peak,
        integration_start_min=left_bound,
        integration_end_min=right_bound,
    )


def summarise_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    aggregations: dict[str, tuple[str, str]] = {
        "metabolites_detected": ("has_peak", "sum"),
        "metabolites_total": ("has_peak", "count"),
        "median_efficiency_plates": ("efficiency_plates", "median"),
        "median_fwhm_min": ("fwhm_min", "median"),
        "median_asymmetry_10": ("asymmetry_10", "median"),
        "median_snr": ("snr", "median"),
    }
    if "is_acceptable_shape" in metrics.columns:
        aggregations["acceptable_shape_count"] = ("is_acceptable_shape", "sum")

    summary = metrics.groupby("column_id", dropna=False).agg(**aggregations).reset_index()
    summary["coverage_pct"] = (
        summary["metabolites_detected"] / summary["metabolites_total"] * 100.0
    )
    if "acceptable_shape_count" in summary.columns:
        summary["acceptable_shape_pct"] = (
            summary["acceptable_shape_count"] / summary["metabolites_total"] * 100.0
        )
    return summary


def rank_columns(
    metrics: pd.DataFrame, selected_metabolites: list[str] | None = None
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    scoped = metrics
    if selected_metabolites:
        scoped = metrics[metrics["metabolite"].isin(selected_metabolites)]
    if scoped.empty:
        return pd.DataFrame()

    grouped = (
        scoped.groupby("column_id", dropna=False)
        .agg(
            detected=("has_peak", "sum"),
            total=("has_peak", "count"),
            median_efficiency=("efficiency_plates", "median"),
            median_fwhm=("fwhm_min", "median"),
            median_asymmetry=("asymmetry_10", "median"),
            median_snr=("snr", "median"),
        )
        .reset_index()
    )
    if "is_acceptable_shape" in scoped.columns:
        acceptable_rate = (
            scoped.groupby("column_id", dropna=False)["is_acceptable_shape"]
            .mean()
            .reset_index(name="acceptable_rate")
        )
        grouped = grouped.merge(acceptable_rate, on="column_id", how="left")

    grouped["coverage"] = grouped["detected"] / grouped["total"].clip(lower=1)

    efficiency_norm = _minmax(grouped["median_efficiency"], invert=False)
    fwhm_norm = _minmax(grouped["median_fwhm"], invert=True)
    asymmetry_penalty = (grouped["median_asymmetry"] - 1.0).abs()
    asymmetry_norm = _minmax(asymmetry_penalty, invert=True)
    snr_norm = _minmax(grouped["median_snr"], invert=False)
    coverage_norm = _minmax(grouped["coverage"], invert=False)
    if "acceptable_rate" in grouped.columns:
        acceptable_norm = _minmax(grouped["acceptable_rate"], invert=False)
        grouped["composite_score"] = (
            0.30 * coverage_norm
            + 0.20 * efficiency_norm
            + 0.15 * fwhm_norm
            + 0.15 * asymmetry_norm
            + 0.10 * snr_norm
            + 0.10 * acceptable_norm
        )
    else:
        grouped["composite_score"] = (
            0.35 * coverage_norm
            + 0.2 * efficiency_norm
            + 0.2 * fwhm_norm
            + 0.15 * asymmetry_norm
            + 0.1 * snr_norm
        )
    return grouped.sort_values("composite_score", ascending=False).reset_index(drop=True)


def assess_peak_acceptability(
    metrics: pd.DataFrame,
    min_snr: float = 5.0,
    max_fwhm_min: float = 0.4,
    min_asymmetry_10: float = 0.7,
    max_asymmetry_10: float = 1.8,
    min_efficiency_plates: float = 1000.0,
) -> pd.DataFrame:
    """Label each run/metabolite peak as acceptable or not based on shape criteria."""
    if metrics.empty:
        return metrics.copy()

    scored = metrics.copy()
    scored["snr_ok"] = scored["snr"].fillna(-np.inf) >= float(min_snr)
    scored["fwhm_ok"] = scored["fwhm_min"].fillna(np.inf) <= float(max_fwhm_min)
    scored["asymmetry_ok"] = scored["asymmetry_10"].between(
        float(min_asymmetry_10), float(max_asymmetry_10), inclusive="both"
    )
    scored["efficiency_ok"] = (
        scored["efficiency_plates"].fillna(-np.inf) >= float(min_efficiency_plates)
    )

    scored["is_acceptable_shape"] = (
        scored["has_peak"]
        & scored["snr_ok"]
        & scored["fwhm_ok"]
        & scored["asymmetry_ok"]
        & scored["efficiency_ok"]
    )

    reasons: list[str] = []
    for _, row in scored.iterrows():
        if not bool(row["has_peak"]):
            reasons.append("no_peak")
            continue
        failed: list[str] = []
        if not bool(row["snr_ok"]):
            failed.append("low_snr")
        if not bool(row["fwhm_ok"]):
            failed.append("wide_peak")
        if not bool(row["asymmetry_ok"]):
            failed.append("tailing/fronting")
        if not bool(row["efficiency_ok"]):
            failed.append("low_efficiency")
        reasons.append("acceptable" if not failed else ",".join(failed))
    scored["acceptability_reason"] = reasons
    return scored


def build_acceptability_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create metabolite x column acceptability (%) matrix."""
    if metrics.empty or "is_acceptable_shape" not in metrics.columns:
        return pd.DataFrame()

    matrix = (
        metrics.groupby(["metabolite", "column_id"], dropna=False)["is_acceptable_shape"]
        .mean()
        .mul(100.0)
        .unstack("column_id")
        .fillna(0.0)
    )
    if matrix.empty:
        return pd.DataFrame()
    return matrix.sort_index()


def _minmax(series: pd.Series, invert: bool) -> pd.Series:
    valid = series.astype(float).replace([np.inf, -np.inf], np.nan)
    if valid.notna().sum() == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    low = float(valid.min(skipna=True))
    high = float(valid.max(skipna=True))
    if np.isclose(low, high):
        normalized = pd.Series(np.ones(len(series)), index=series.index, dtype=float)
    else:
        normalized = (valid - low) / (high - low)
        normalized = normalized.fillna(0.0)
    if invert:
        normalized = 1.0 - normalized
    return normalized
