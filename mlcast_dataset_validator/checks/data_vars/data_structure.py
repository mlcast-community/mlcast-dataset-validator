from typing import Sequence

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.3"


@log_function_call
def check_data_structure(
    ds: xr.Dataset,
    *,
    dim_order: Sequence[str],
    allowed_dtypes: Sequence[str],
) -> ValidationReport:
    """Check data structure requirements."""
    report = ValidationReport()
    # Find all grid_mapping data variables since we don't want to check those
    grid_mapping_vars = set()
    for var in ds.data_vars:
        if "grid_mapping" in ds[var].attrs:
            grid_mapping_vars.add(ds[var].attrs["grid_mapping"])
    data_vars = set(ds.data_vars) - grid_mapping_vars

    for data_var in data_vars:
        data_array = ds[data_var]

        # Check dimension order
        if tuple(data_array.dims) == tuple(dim_order):
            report.add(
                SECTION_ID,
                f"Dimension order for {data_var}",
                "PASS",
                f"Dimension order matches {dim_order}",
            )
        else:
            report.add(
                SECTION_ID,
                f"Dimension order for {data_var}",
                "FAIL",
                f"Expected dimension order {dim_order}, found {data_array.dims}",
            )

        # Check data type
        if str(data_array.dtype) in allowed_dtypes:
            report.add(
                SECTION_ID,
                f"Data type for {data_var}",
                "PASS",
                f"Data type '{data_array.dtype}' is allowed",
            )
        else:
            report.add(
                SECTION_ID,
                f"Data type for {data_var}",
                "FAIL",
                f"Data type '{data_array.dtype}' is not allowed. Allowed types: {allowed_dtypes}",
            )

    return report
