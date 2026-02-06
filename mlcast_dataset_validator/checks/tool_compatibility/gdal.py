"""
GDAL compatibility checks derived from the standalone tooling script supplied
in the radar source validator project.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Optional

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from ..data_vars_filter import select_data_var
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.1"

try:  # pragma: no cover - optional dependency handling
    from osgeo import gdal, osr  # type: ignore

    gdal.UseExceptions()
    GDAL_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback
    GDAL_AVAILABLE = False
    gdal = None  # type: ignore
    osr = None  # type: ignore

try:  # pragma: no cover - optional dependency handling
    import rioxarray  # noqa: F401

    RIOXARRAY_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback
    RIOXARRAY_AVAILABLE = False


def _select_data_variable(ds: xr.Dataset, preferred: Optional[str]) -> Optional[str]:
    """
    Choose the data variable that will be exported for GDAL testing.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset that will be validated.
    preferred : str or None
        Caller-provided variable name to prioritize.

    Returns
    -------
    Optional[str]
        Name of the data variable to use, or None if unavailable.
    """
    return select_data_var(ds, preferred, require_grid_mapping=True)


def _prepare_sample_slice(data_array: xr.DataArray) -> xr.DataArray:
    """
    Create a 2D sample slice suitable for export to GeoTIFF.

    Parameters
    ----------
    data_array : xr.DataArray
        Source data array from the dataset.

    Returns
    -------
    xr.DataArray
        A slice with dimensions ordered as (y, x).
    """
    sample = data_array
    if "time" in sample.dims:
        sample = sample.isel(time=0, drop=True)
    sample = sample.squeeze(drop=True)

    spatial_dims = [dim for dim in sample.dims if dim not in {"time", "t"}]
    if len(spatial_dims) < 2:
        raise ValueError("Need at least two spatial dimensions to export raster data.")

    # Rename final two dims to y/x for rioxarray compatibility
    rename_map: Dict[str, str] = {}
    if spatial_dims[-2] != "y":
        rename_map[spatial_dims[-2]] = "y"
    if spatial_dims[-1] != "x":
        rename_map[spatial_dims[-1]] = "x"

    if rename_map:
        sample = sample.rename(rename_map)

    return sample


def _cleanup_temp_file(path: Path) -> None:
    """Remove a temporary file, suppressing any filesystem errors."""
    try:
        path.unlink(missing_ok=True)
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


@log_function_call
def check_gdal_compatibility(
    ds: xr.Dataset,
    *,
    data_variable: Optional[str] = None,
) -> ValidationReport:
    """
    Validate GDAL tool compatibility for a dataset.

    The check is inspired by the standalone GDAL validation script and verifies
    three core behaviors:
      1. The CRS can be parsed by GDAL from the stored WKT string.
      2. The identified CRS indicates whether the dataset is projected.
      3. Exporting a sample slice via rioxarray and roundtripping through GDAL
         succeeds (if rioxarray is available).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset opened via xarray.
    data_variable : str, optional
        Specific data variable to export; defaults to the first data variable.

    Returns
    -------
    ValidationReport
        Detailed PASS/FAIL/WARNING entries describing GDAL compatibility.
    """

    report = ValidationReport()

    if not GDAL_AVAILABLE:
        report.add(
            SECTION_ID,
            "GDAL availability",
            "WARNING",
            "GDAL bindings are not installed; skipping compatibility checks.",
        )
        return report

    target_var = _select_data_variable(ds, data_variable)
    if target_var is None:
        report.add(
            SECTION_ID,
            "Data variable selection",
            "FAIL",
            "No data variable with a 'grid_mapping' attribute is available for GDAL checks.",
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
            "GDAL WKT parsing",
            "FAIL",
            f"CRS variable '{grid_mapping_name}' is missing 'crs_wkt' metadata.",
        )
        return report

    try:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt_string)
        if srs.IsProjected():
            detail = f"Projected CRS detected ({srs.GetAttrValue('PROJECTION') or 'unknown projection'})."
        elif srs.IsGeographic():
            detail = "Geographic CRS detected (latitude/longitude)."
        else:
            detail = (
                "CRS parsed but type is ambiguous (neither projected nor geographic)."
            )

        report.add(
            SECTION_ID,
            "GDAL WKT parsing",
            "PASS",
            detail,
        )
    except Exception as exc:
        report.add(
            SECTION_ID,
            "GDAL WKT parsing",
            "FAIL",
            f"GDAL failed to parse CRS WKT: {exc}",
        )
        return report

    if not RIOXARRAY_AVAILABLE:
        report.add(
            SECTION_ID,
            "GDAL roundtrip",
            "WARNING",
            "rioxarray is not installed; skipping GeoTIFF export roundtrip.",
        )
        return report

    try:
        sample = _prepare_sample_slice(data_array)
        sample = sample.rio.write_crs(wkt_string, inplace=False)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            sample.rio.to_raster(tmp_path)
            gdal_ds = gdal.Open(tmp_path.as_posix())
            if gdal_ds is None:
                raise RuntimeError("GDAL could not open the exported GeoTIFF.")

            geo_transform = gdal_ds.GetGeoTransform(can_return_null=True)
            projection = gdal_ds.GetProjectionRef()
            gdal_ds = None  # Close dataset

            if geo_transform and projection:
                report.add(
                    SECTION_ID,
                    "GDAL roundtrip",
                    "PASS",
                    "GeoTIFF export via rioxarray can be read back by GDAL with geotransform/projection metadata.",
                )
            else:
                report.add(
                    SECTION_ID,
                    "GDAL roundtrip",
                    "FAIL",
                    "Roundtrip succeeded but GDAL reported missing geotransform or projection metadata.",
                )
        finally:
            _cleanup_temp_file(tmp_path)
    except Exception as exc:
        report.add(
            SECTION_ID,
            "GDAL roundtrip",
            "FAIL",
            f"GeoTIFF export/read via GDAL failed: {exc}",
        )

    return report
