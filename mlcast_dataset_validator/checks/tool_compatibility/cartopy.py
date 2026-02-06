"""
Cartopy compatibility checks derived from the standalone tooling script supplied
alongside the radar source validator project.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from ..data_vars_filter import select_data_var
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.2"

try:  # pragma: no cover - optional dependency handling
    import cartopy.crs as ccrs  # type: ignore

    CARTOPY_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback
    CARTOPY_AVAILABLE = False
    ccrs = None  # type: ignore


def _select_data_variable(ds: xr.Dataset, preferred: Optional[str]) -> Optional[str]:
    """Choose a data variable to use for sampling."""
    return select_data_var(ds, preferred, require_grid_mapping=True)


@log_function_call
def check_cartopy_compatibility(
    ds: xr.Dataset,
    *,
    data_variable: Optional[str] = None,
) -> ValidationReport:
    """
    Validate Cartopy compatibility for a dataset.

    The check ensures that:
      1. Cartopy can instantiate a CRS object from the stored WKT string.
      2. The CRS description includes a BBOX (recommended for plotting).
      3. Coordinate bounds exist and can be transformed to PlateCarree.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset opened via xarray.
    data_variable : str, optional
        Specific data variable to inspect; defaults to the first data variable.

    Returns
    -------
    ValidationReport
        Detailed PASS/FAIL/WARNING entries describing Cartopy compatibility.
    """
    report = ValidationReport()

    if not CARTOPY_AVAILABLE:
        report.add(
            SECTION_ID,
            "Cartopy availability",
            "WARNING",
            "cartopy is not installed; skipping cartopy-specific checks.",
        )
        return report

    target_var = _select_data_variable(ds, data_variable)
    if target_var is None:
        report.add(
            SECTION_ID,
            "Data variable selection",
            "FAIL",
            "No data variable with a 'grid_mapping' attribute is available for cartopy checks.",
        )
        return report

    data_array = ds[target_var]
    grid_mapping_name = data_array.attrs.get("grid_mapping")
    if not grid_mapping_name or grid_mapping_name not in ds.variables:
        report.add(
            SECTION_ID,
            "CRS metadata",
            "FAIL",
            f"Data variable '{target_var}' lacks a valid 'grid_mapping' attribute.",
        )
        return report

    crs_var = ds[grid_mapping_name]
    wkt_string = crs_var.attrs.get("crs_wkt")
    if not wkt_string:
        report.add(
            SECTION_ID,
            "Cartopy CRS creation",
            "FAIL",
            f"CRS variable '{grid_mapping_name}' is missing 'crs_wkt' metadata.",
        )
        return report

    try:
        crs = ccrs.CRS(wkt_string)
        report.add(
            SECTION_ID,
            "Cartopy CRS creation",
            "PASS",
            f"Successfully created cartopy CRS instance ({type(crs).__name__}).",
        )
    except Exception as exc:
        report.add(
            SECTION_ID,
            "Cartopy CRS creation",
            "FAIL",
            f"Cartopy failed to parse CRS WKT: {exc}",
        )
        return report

    lower_wkt = wkt_string.lower()
    if "bbox" in lower_wkt:
        report.add(
            SECTION_ID,
            "Cartopy BBOX check",
            "PASS",
            "WKT definition includes a BBOX, aiding cartopy plotting.",
        )
    else:
        report.add(
            SECTION_ID,
            "Cartopy BBOX check",
            "WARNING",
            "WKT lacks a BBOX definition; cartopy plots may require manual extents.",
        )

    # Attempt to transform a handful of coordinates into PlateCarree for sanity.
    if {"x", "y"}.issubset(ds.coords):
        try:
            x_vals = ds.x.values
            y_vals = ds.y.values
            if len(x_vals) == 0 or len(y_vals) == 0:
                raise ValueError("Empty coordinate arrays.")

            x_sample = x_vals[:: max(1, len(x_vals) // 5)][:5]
            y_sample = y_vals[:: max(1, len(y_vals) // 5)][:5]
            xx, yy = np.meshgrid(x_sample, y_sample)

            plate_carree = ccrs.PlateCarree()
            transformed = plate_carree.transform_points(crs, xx.flatten(), yy.flatten())

            if np.isnan(transformed).any():
                report.add(
                    SECTION_ID,
                    "Cartopy coordinate transform",
                    "WARNING",
                    "Coordinate transformation produced NaN values.",
                )
            else:
                report.add(
                    SECTION_ID,
                    "Cartopy coordinate transform",
                    "PASS",
                    f"Successfully transformed {transformed.shape[0]} coordinate pairs to PlateCarree.",
                )
        except Exception as exc:
            report.add(
                SECTION_ID,
                "Cartopy coordinate transform",
                "FAIL",
                f"Failed to transform coordinates for cartopy plotting: {exc}",
            )
    else:
        report.add(
            SECTION_ID,
            "Cartopy coordinate transform",
            "WARNING",
            "Dataset lacks 'x' and 'y' coordinates; skipping transform test.",
        )

    return report
