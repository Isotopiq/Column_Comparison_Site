# Column Comparison Site

Streamlit application for comparing chromatographic column performance using standard-metabolite mzXML runs (OpenMS / pyOpenMS backend).

## What this app does

- Upload **multiple mzXML files** from different columns/runs.
- Upload a **metabolite list** (name + target m/z, optional expected RT).
- Extract EICs for each metabolite with configurable m/z tolerance.
- Compare **peak shape side-by-side by column**.
- Calculate peak metrics:
  - Apex RT
  - Peak area
  - FWHM
  - Asymmetry factor (10% height)
  - Estimated theoretical plates
  - Approximate S/N
- Save standard apex RTs into a persistent **retention-time library** (`data/retention_library.json`).
- Recommend columns based on selected target metabolites using:
  - Current-run composite quality score
  - Saved-library metabolite coverage + RT stability

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Input format

### Metabolite table (CSV or Excel)

Required:
- metabolite name column (e.g. `metabolite`)
- m/z column (e.g. `mz`)

Optional:
- expected RT column in minutes (e.g. `expected_rt_min`)

Example:

| metabolite | mz       | expected_rt_min |
|------------|----------|-----------------|
| Lactate    | 89.0244  | 2.3             |
| Pyruvate   | 87.0088  | 2.9             |

### mzXML files

Upload one or more standard runs. In the app you can edit:
- `run_id` (replicate/run label)
- `column_id` (column grouping label for side-by-side comparison)

## Notes

- The app currently focuses on **MS1 EIC extraction** for targeted metabolites.
- RT library matching uses exact metabolite name matches.
- Saved library is local JSON and can be downloaded/uploaded manually as needed.
