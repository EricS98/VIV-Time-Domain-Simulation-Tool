# reporting/excel.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

def _row_from_single(
    structure: Any,
    config: Any,
    fd: Optional[Any] = None,
    td: Optional[Any] = None,
) -> Dict[str, Any]:
    
    row: Dict[str, Any] = {}

    row.update(structure.get_summary())

    d = getattr(structure, "diameter", None)

    # Frequency-domain
    if fd is not None:
        row.update({
            "fd.sigma_y": fd.sigma_y,
            "fd.peak_factor": fd.peak_factor,
            "fd.y_max": fd.y_max,
            "fd.Ka_r_n": fd.Ka_r_n,
            "fd.delta_a": fd.delta_a,
            "fd.delta_total": fd.delta_total,
            "fd.converged": fd.converged,
        })
        if d:
            row["fd.y_max_over_d"] = fd.y_max / d
            row["fd.sigma_y_over_d"] = fd.sigma_y / d

    # Time-domain
    if td is not None:
        row.update({
            "td.sigma_y": td.sigma_y,
            "td.peak_factor": td.peak_factor,
            "td.y_max": td.y_max,
            "td.steady_start_time": td.steady_start_time,
        })
        if d:
            row["td.y_max_over_d"] = td.y_max / d
            row["td.sigma_y_over_d"] = td.sigma_y / d

        # Grid-info
        if getattr(td, "grid", None) is not None:
            gi = td.grid.get_info()
            row.update({
                "td.dt": gi.get("time_step"),
                "td.T": gi.get("duration"),
                "td.N": gi.get("n_samples"),
                "td.df": gi.get("frequency_resolution"),
                "td.f_nyquist": gi.get("nyquist_frequency"),
            })

    return row

def export_single_summary_xlsx(
    output_path: Path,
    structure: Any,
    config: Any,
    result: Dict[str, Any],
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd = None
    if result.get("frequency") and result["frequency"].get("response"):
        fd = result["frequency"]["response"]

    td = None
    if result.get("time") and result["time"].get("time_domain"):
        td = result["time"]["time_domain"]

    df = pd.DataFrame([_row_from_single(structure, config, fd=fd, td=td)])

    # Column ordering
    preferred = [
        "name",
        "height", "diameter", "equivalent_mass", "f_n", "delta_s",
        "fd.sigma_y_over_d", "fd.y_max_over_d", "fd.peak_factor",
        "td.sigma_y_over_d", "td.y_max_over_d", "td.peak_factor",
    ]

    cols = [c for c in preferred if c in df.columns] \
        + [c for c in df.columns if c not in preferred]
    df = df[cols]

    # Rounding
    round_map = {
        "height": 2,
        "diameter": 3,
        "f_n": 3,
        "fd.sigma_y_over_d": 4,
        "fd.y_max_over_d": 4,
        "td.sigma_y_over_d": 4,
        "td.y_max_over_d": 4,
    }

    for col, nd in round_map.items():
        if col in df.columns:
            df[col] = df[col].round(nd)

    # Units
    unit_map = {
        "height": "m",
        "diameter": "m",
        "equivalent_mass": "kg/m",
        "f_n": "Hz",
        "delta_s": "-",
        "fd.sigma_y_over_d": "-",
        "fd.y_max_over_d": "-",
        "td.sigma_y_over_d": "-",
        "td.y_max_over_d": "-",
        "td.dt": "s",
        "td.T": "s",
        "td.df": "Hz",
        "td.f_nyquist": "Hz",
    }
    df = _insert_units_row(df, unit_map)

    df.to_excel(output_path, index=False, sheet_name="Summary")
    return output_path

def export_multiple_summary_xlsx(
    output_path: Path,
    structures: List[Any],
    template_config: Any,
    results: Dict[str, Any],
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd_list = results.get("frequency_domain") or []
    td_list = results.get("time_domain") or []

    rows = []
    for i, s in enumerate(structures):
        fd = fd_list[i]["response"] if i < len(fd_list) and fd_list[i].get("response") else None
        td = td_list[i]["time_domain"] if i < len(td_list) and td_list[i].get("time_domain") else None
        rows.append(_row_from_single(s, template_config, fd=fd, td=td))

    df = pd.DataFrame(rows)

    # Column ordering
    preferred = [
        "name", "No",
        "height", "diameter", "equivalent_mass", "f_n", "delta_s",
        "fd.sigma_y_over_d", "fd.y_max_over_d", "fd.peak_factor", "fd.converged",
        "td.sigma_y_over_d", "td.y_max_over_d", "td.peak_factor", "td.steady_start_time",
    ]

    cols = [c for c in preferred if c in df.columns] \
        + [c for c in df.columns if c not in preferred]
    df = df[cols]

    # Rounding
    round_map = {
        "height": 2,
        "diameter": 3,
        "f_n": 3,
        "fd.sigma_y_over_d": 4,
        "fd.y_max_over_d": 4,
        "td.sigma_y_over_d": 4,
        "td.y_max_over_d": 4,
    }

    for col, nd in round_map.items():
        if col in df.columns:
            df[col] = df[col].round(nd)

    # Units
    unit_map = {
        "height": "m",
        "diameter": "m",
        "equivalent_mass": "kg/m",
        "f_n": "Hz",
        "delta_s": "-",
        "fd.sigma_y_over_d": "-",
        "fd.y_max_over_d": "-",
        "td.sigma_y_over_d": "-",
        "td.y_max_over_d": "-",
        "td.dt": "s",
        "td.T": "s",
        "td.df": "Hz",
        "td.f_nyquist": "Hz",
    }
    df = _insert_units_row(df, unit_map)

    df.to_excel(output_path, index=False, sheet_name="Summary_All")
    return output_path

def _insert_units_row(df: pd.DataFrame, unit_map: dict) -> pd.DataFrame:
    units = {c: unit_map.get(c, "") for c in df.columns}
    df2 = pd.concat([pd.DataFrame([units]), df], ignore_index=True)
    return df2