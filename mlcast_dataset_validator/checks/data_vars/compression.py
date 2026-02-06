from typing import Sequence

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.2"


def get_compressor_name(da: xr.DataArray) -> str:
    """
    Return the name of the compressor used for a Zarr-backed xarray.DataArray.

    This function handles both Zarr v2 datasets written directly by xarray/zarr
    (where compression appears in `da.encoding["compressor"]`) and datasets
    converted from NetCDF4/HDF5 sources (where compression is stored in
    `da.encoding["filters"]` as e.g. `Zlib(level=5)`).

    The distinction arises because:
      - Zarr natively stores a single 'compressor' codec (e.g. Blosc, Zstd, etc.).
      - HDF5-based formats (NetCDF4) instead use a *pipeline* of filters,
        where the final step may itself be a compressor (e.g. Shuffle + Zlib).

    Parameters
    ----------
    da : xarray.DataArray
        A DataArray opened from a Zarr or NetCDF source.

    Returns
    -------
    str
        Name of the compressor (e.g. "zstd", "zlib", "blosc") or "none"
        if no compression is applied.
    """
    enc = getattr(da, "encoding", {}) or {}

    # First, check for native Zarr compressor
    comp = enc.get("compressor") or enc.get("compressors")

    def _extract_name(obj) -> str | None:
        if obj is None:
            return None
        if isinstance(obj, (list, tuple)):
            for item in obj:
                name = _extract_name(item)
                if name:
                    return name
            return None
        if isinstance(obj, str):
            return obj.lower()
        name = getattr(obj, "codec_id", None)
        if isinstance(name, str):
            return name.lower()
        if name:
            return str(name).lower()
        return obj.__class__.__name__.lower()

    # Handle a single or nested compressor instance
    name = _extract_name(comp)
    if name and name != "tuple":
        return name

    # Otherwise, fall back to inspecting filters (NetCDF/HDF5-style)
    filters = enc.get("filters") or []
    for f in filters:
        name = _extract_name(f)
        if any(
            name and k in name
            for k in ("zlib", "gzip", "bz2", "blosc", "zstd", "lz4", "snappy")
        ):
            return name

    # No compression found
    return None


@log_function_call
def check_compression(
    ds: xr.Dataset,
    *,
    require_compression: bool,
    recommended_compression: str,
    allow_coord_algs: Sequence[str],
) -> ValidationReport:
    """Check compression requirements."""
    report = ValidationReport()

    # Find all grid_mapping data variables since we don't want to check those
    grid_mapping_vars = set()
    for var in ds.data_vars:
        if "grid_mapping" in ds[var].attrs:
            grid_mapping_vars.add(ds[var].attrs["grid_mapping"])
    data_vars = set(ds.data_vars) - grid_mapping_vars

    for data_var in data_vars:
        da = ds[data_var]
        compressor = get_compressor_name(da)
        report_title = f"DataArray compression {da.name}"

        if require_compression and compressor is None:
            report.add(
                SECTION_ID,
                report_title,
                "FAIL",
                f"{da.name} DataArray does not use compression",
            )
            continue

        if compressor == recommended_compression:
            report.add(
                SECTION_ID,
                report_title,
                "PASS",
                f"{da.name} DataArray uses recommended compression: {recommended_compression}",
            )
        elif compressor is None:
            # Compression not required and none present
            continue
        else:
            report.add(
                SECTION_ID,
                report_title,
                "WARNING",
                f"{da.name} DataArrays uses compression: {compressor}, "
                f"recommended is {recommended_compression}",
            )

    return report
