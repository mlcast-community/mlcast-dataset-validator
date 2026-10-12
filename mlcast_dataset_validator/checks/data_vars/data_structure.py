from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from ..data_vars_filter import iter_data_vars
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


@log_function_call
def check_variable_dims_and_dtype(
    ds: xr.Dataset,
    *,
    required_dims: Sequence[str],
    dtype_kinds: str = "f",
    require_finite_along: str | None = None,
) -> ValidationReport:
    """
    Check that every data variable spans the required dimensions (in any
    order) and has a dtype of an allowed kind.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.
    required_dims : Sequence[str]
        Dimensions every data variable MUST have. Other dimensions are allowed.
    dtype_kinds : str, optional
        numpy dtype kind characters that are allowed, "f" (floating point) by
        default.
    require_finite_along : str, optional
        If given, every entry along this dimension SHOULD have at least one
        finite value in each data variable (a WARNING otherwise).

    Returns
    -------
    ValidationReport
        Report with dimension and dtype results for every data variable.
    """
    report = ValidationReport()

    for name, data_array in iter_data_vars(ds):
        missing = [dim for dim in required_dims if dim not in data_array.dims]
        if missing:
            report.add(
                SECTION_ID,
                f"Dimensions of '{name}'",
                "FAIL",
                f"Missing required dimension(s) {missing}, found {data_array.dims}",
            )
        else:
            report.add(
                SECTION_ID,
                f"Dimensions of '{name}'",
                "PASS",
                f"Spans required dimensions {list(required_dims)}",
            )

        if data_array.dtype.kind in dtype_kinds:
            report.add(
                SECTION_ID,
                f"Data type of '{name}'",
                "PASS",
                f"Data type '{data_array.dtype}' is allowed",
            )
        else:
            report.add(
                SECTION_ID,
                f"Data type of '{name}'",
                "FAIL",
                f"Data type '{data_array.dtype}' is not of an allowed kind "
                f"({dtype_kinds!r})",
            )

        if require_finite_along and require_finite_along in data_array.dims:
            other_dims = [d for d in data_array.dims if d != require_finite_along]
            has_finite = np.isfinite(data_array).any(dim=other_dims).compute()
            empty = list(data_array[require_finite_along].values[~has_finite.values])
            if empty:
                report.add(
                    SECTION_ID,
                    f"Finite values of '{name}'",
                    "WARNING",
                    f"No finite values for {require_finite_along} {empty}",
                )
            else:
                report.add(
                    SECTION_ID,
                    f"Finite values of '{name}'",
                    "PASS",
                    f"Every {require_finite_along} has finite values",
                )

    return report
