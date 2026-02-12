from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def default_rt_library() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": _timestamp(),
        "columns": {},
    }


def load_rt_library(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        return default_rt_library()

    with path_obj.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        return default_rt_library()
    if "columns" not in raw or not isinstance(raw["columns"], dict):
        raw = default_rt_library()
    return raw


def save_rt_library(path: str | Path, library: dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    library["updated_at"] = _timestamp()
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, sort_keys=True)


def update_rt_library_from_metrics(
    library: dict[str, Any],
    metrics: pd.DataFrame,
    min_snr: float = 3.0,
    max_points_per_metabolite: int = 100,
) -> dict[str, Any]:
    if metrics.empty:
        return library

    valid = metrics.copy()
    valid = valid[valid["has_peak"] == True]  # noqa: E712 - explicit bool comparison
    valid = valid[valid["apex_rt_min"].notna()]
    if "snr" in valid.columns:
        valid = valid[(valid["snr"].isna()) | (valid["snr"] >= min_snr)]
    if valid.empty:
        return library

    columns = library.setdefault("columns", {})

    for _, row in valid.iterrows():
        column_id = str(row["column_id"])
        metabolite = str(row["metabolite"])
        column_bucket = columns.setdefault(column_id, {"metabolites": {}})
        metabolite_bucket = column_bucket["metabolites"].setdefault(
            metabolite,
            {
                "mz": float(row["mz"]),
                "rt_values_min": [],
                "source_runs": [],
                "last_updated": _timestamp(),
            },
        )

        rt_values = metabolite_bucket.setdefault("rt_values_min", [])
        run_ids = metabolite_bucket.setdefault("source_runs", [])
        rt_values.append(float(row["apex_rt_min"]))
        run_ids.append(str(row["run_id"]))

        if len(rt_values) > max_points_per_metabolite:
            metabolite_bucket["rt_values_min"] = rt_values[-max_points_per_metabolite:]
            metabolite_bucket["source_runs"] = run_ids[-max_points_per_metabolite:]

        # Recompute stable summary stats for easy matching.
        array = np.asarray(metabolite_bucket["rt_values_min"], dtype=float)
        metabolite_bucket["median_rt_min"] = float(np.median(array))
        metabolite_bucket["std_rt_min"] = float(np.std(array))
        metabolite_bucket["n_observations"] = int(array.size)
        metabolite_bucket["last_updated"] = _timestamp()
        metabolite_bucket["mz"] = float(row["mz"])

    library["updated_at"] = _timestamp()
    return library


def rt_library_to_dataframe(library: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column_id, column_payload in library.get("columns", {}).items():
        metabolites = column_payload.get("metabolites", {})
        for metabolite, payload in metabolites.items():
            rows.append(
                {
                    "column_id": column_id,
                    "metabolite": metabolite,
                    "mz": payload.get("mz"),
                    "median_rt_min": payload.get("median_rt_min"),
                    "std_rt_min": payload.get("std_rt_min"),
                    "n_observations": payload.get("n_observations", 0),
                    "last_updated": payload.get("last_updated"),
                }
            )
    return pd.DataFrame(rows)


def merge_expected_rts(
    metabolite_table: pd.DataFrame,
    column_id: str,
    library: dict[str, Any],
    metabolite_col: str = "metabolite",
) -> pd.DataFrame:
    updated = metabolite_table.copy()
    if metabolite_col not in updated.columns:
        return updated

    expected_lookup: dict[str, float] = {}
    column_bucket = library.get("columns", {}).get(column_id, {})
    metabolites = column_bucket.get("metabolites", {})
    for metabolite_name, payload in metabolites.items():
        if payload.get("median_rt_min") is None:
            continue
        expected_lookup[metabolite_name] = float(payload["median_rt_min"])

    updated["expected_rt_from_library_min"] = (
        updated[metabolite_col].astype(str).map(expected_lookup)
    )
    return updated
