from __future__ import annotations

import io
import os
import tempfile
import zipfile
from datetime import UTC, datetime
from typing import Any

import pandas as pd


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def figure_to_png_bytes(
    figure: Any, width: int = 1400, height: int = 800, scale: int = 2
) -> bytes | None:
    """Best-effort conversion of a Plotly figure to PNG bytes."""
    try:
        return figure.to_image(format="png", width=width, height=height, scale=scale)
    except Exception:
        return None


def build_metabolite_report_html(
    metabolite: str,
    metabolite_metrics: pd.DataFrame,
    peak_shape_figure: Any,
    note: str | None = None,
    settings: dict[str, Any] | None = None,
) -> bytes:
    settings_rows = ""
    if settings:
        settings_rows = "".join(
            f"<tr><td><b>{key}</b></td><td>{value}</td></tr>"
            for key, value in settings.items()
        )

    summary_by_column = (
        metabolite_metrics.groupby("column_id", dropna=False)
        .agg(
            runs=("run_id", "count"),
            accepted_shape=("is_acceptable_shape", "sum")
            if "is_acceptable_shape" in metabolite_metrics.columns
            else ("has_peak", "sum"),
            median_apex_rt=("apex_rt_min", "median"),
            median_fwhm=("fwhm_min", "median"),
            median_asymmetry=("asymmetry_10", "median"),
            median_efficiency=("efficiency_plates", "median"),
            median_snr=("snr", "median"),
        )
        .reset_index()
    )

    metrics_html = metabolite_metrics.to_html(
        index=False,
        classes="table",
        border=0,
        justify="left",
    )
    summary_html = summary_by_column.to_html(
        index=False,
        classes="table",
        border=0,
        justify="left",
    )
    figure_html = peak_shape_figure.to_html(full_html=False, include_plotlyjs="cdn")

    note_block = note.strip() if note else "No note saved."
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Metabolite Report - {metabolite}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #1f2937; }}
    h1, h2 {{ margin: 0 0 10px 0; }}
    .meta {{ margin-bottom: 18px; }}
    .panel {{ margin-bottom: 20px; padding: 12px; border: 1px solid #d1d5db; border-radius: 8px; }}
    table.table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    table.table td, table.table th {{ border: 1px solid #e5e7eb; padding: 6px; text-align: left; }}
    table.table th {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <h1>Metabolite comparison report: {metabolite}</h1>
  <div class="meta">Generated at: {_timestamp()}</div>

  <div class="panel">
    <h2>Saved note</h2>
    <div>{note_block}</div>
  </div>

  <div class="panel">
    <h2>Analysis settings</h2>
    <table class="table">
      <tbody>{settings_rows}</tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Peak shape</h2>
    {figure_html}
  </div>

  <div class="panel">
    <h2>Column summary</h2>
    {summary_html}
  </div>

  <div class="panel">
    <h2>Run-level metrics</h2>
    {metrics_html}
  </div>
</body>
</html>
"""
    return html.encode("utf-8")


def build_metabolite_report_pdf(
    metabolite: str,
    metabolite_metrics: pd.DataFrame,
    peak_shape_figure: Any,
    note: str | None = None,
    settings: dict[str, Any] | None = None,
) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError as exc:  # pragma: no cover - depends on runtime install
        raise RuntimeError(
            "PDF export requires fpdf2. Install dependencies from requirements.txt."
        ) from exc

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, f"Metabolite report: {metabolite}", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"Generated: {_timestamp()}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 7, "Saved note", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(
        0,
        5,
        note.strip() if note else "No note saved.",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)

    if settings:
        pdf.set_font("Helvetica", style="B", size=12)
        pdf.cell(0, 7, "Settings", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        for key, value in settings.items():
            pdf.multi_cell(0, 5, f"{key}: {value}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    image_bytes = figure_to_png_bytes(peak_shape_figure)
    if image_bytes is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        pdf.set_font("Helvetica", style="B", size=12)
        pdf.cell(0, 7, "Peak shape", new_x="LMARGIN", new_y="NEXT")
        pdf.image(tmp_path, w=190)
        pdf.ln(2)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    preview = metabolite_metrics.copy()
    preview = preview[
        [
            "column_id",
            "run_id",
            "apex_rt_min",
            "fwhm_min",
            "asymmetry_10",
            "efficiency_plates",
            "snr",
            "peak_area",
        ]
    ].sort_values(["column_id", "run_id"])
    preview = preview.head(40).reset_index(drop=True)

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 7, "Run-level metrics (first 40 rows)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Courier", size=8)
    header = "column | run | rt | fwhm | asym | plates | snr | area"
    pdf.multi_cell(0, 4, header, new_x="LMARGIN", new_y="NEXT")
    for _, row in preview.iterrows():
        line = (
            f"{_trim(row['column_id'], 16)} | {_trim(row['run_id'], 24)} | "
            f"{_fmt(row['apex_rt_min'])} | {_fmt(row['fwhm_min'])} | {_fmt(row['asymmetry_10'])} | "
            f"{_fmt(row['efficiency_plates'])} | {_fmt(row['snr'])} | {_fmt(row['peak_area'])}"
        )
        pdf.multi_cell(0, 4, line, new_x="LMARGIN", new_y="NEXT")

    output = pdf.output(dest="S")
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    return output.encode("latin-1")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _trim(value: Any, max_len: int) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_preview_image_bundle(figures: dict[str, Any]) -> dict[str, bytes]:
    """Render available figure previews as PNG bytes."""
    images: dict[str, bytes] = {}
    for name, figure in figures.items():
        if figure is None:
            continue
        png = figure_to_png_bytes(figure, width=1200, height=700, scale=2)
        if png is None:
            continue
        images[f"{name}.png"] = png
    return images


def zip_named_bytes(named_bytes: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in named_bytes.items():
            zf.writestr(name, payload)
    return buffer.getvalue()
