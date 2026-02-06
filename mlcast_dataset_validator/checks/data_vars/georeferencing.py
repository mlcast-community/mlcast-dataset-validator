from typing import Sequence

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.5"


@log_function_call
def check_georeferencing(
    ds: xr.Dataset,
    *,
    require_geozarr: bool,
    require_grid_mapping: bool,
    crs_attrs: Sequence[str],
    require_bbox: bool,
) -> ValidationReport:
    """Check georeferencing requirements."""
    report = ValidationReport()
    # Find all grid_mapping data variables since we don't want to check those
    grid_mapping_vars = set()
    for var in ds.data_vars:
        if "grid_mapping" in ds[var].attrs:
            grid_mapping_vars.add(ds[var].attrs["grid_mapping"])
    data_vars = set(ds.data_vars) - grid_mapping_vars

    for data_var in data_vars:
        data_array = ds[data_var]
        if require_grid_mapping and "grid_mapping" not in data_array.attrs:
            report.add(
                SECTION_ID,
                f"Grid mapping for {data_var}",
                "FAIL",
                f"Data variable '{data_var}' is missing 'grid_mapping' attribute",
            )
            continue

        grid_mapping = data_array.attrs.get("grid_mapping", None)
        if grid_mapping and grid_mapping in ds.variables:
            crs_var = ds[grid_mapping]
            missing_attrs = [attr for attr in crs_attrs if attr not in crs_var.attrs]
            if missing_attrs:
                report.add(
                    SECTION_ID,
                    f"CRS attributes for {data_var}",
                    "FAIL",
                    f"CRS variable '{grid_mapping}' is missing attributes: {missing_attrs}",
                )
            else:
                report.add(
                    SECTION_ID,
                    f"CRS attributes for {data_var}",
                    "PASS",
                    f"CRS variable '{grid_mapping}' has all required attributes",
                )
        else:
            report.add(
                SECTION_ID,
                f"Grid mapping for {data_var}",
                "FAIL",
                f"Data variable '{data_var}' references a non-existent grid mapping variable",
            )

    return report
