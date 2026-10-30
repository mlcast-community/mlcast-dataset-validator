from typing import Optional, Sequence

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from ..data_vars_filter import select_data_vars_by_role
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.3"


@log_function_call
def check_data_structure(
    ds: xr.Dataset,
    *,
    dim_order: Sequence[str],
    allowed_dtypes: Sequence[str],
    use_encoded_dtype: bool = False,
    roles: Optional[Sequence[str]] = None,
) -> ValidationReport:
    """
    Check data structure requirements.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose data variables (other than grid mapping definitions)
        should be evaluated.
    dim_order : Sequence[str]
        Exact dimension order every data variable must have.
    allowed_dtypes : Sequence[str]
        Data types every data variable may have.
    use_encoded_dtype : bool, optional
        Check the on-disk dtype (``encoding["dtype"]``) instead of the decoded
        dtype, e.g. for integer counts that xarray decodes to float because a
        fill value is set. Falls back to the decoded dtype when no encoding is
        available. Defaults to False.
    roles : Sequence[str], optional
        Only check the data variables with these roles (see
        :func:`~mlcast_dataset_validator.checks.data_vars_filter.classify_data_vars`).
        By default all data variables are checked.

    Returns
    -------
    ValidationReport
        A report with dimension order and data type entries per variable.
    """
    report = ValidationReport()
    if roles is not None:
        ds = select_data_vars_by_role(ds, roles)
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
        dtype = data_array.dtype
        if use_encoded_dtype:
            dtype = data_array.encoding.get("dtype", dtype)
        if str(dtype) in allowed_dtypes:
            report.add(
                SECTION_ID,
                f"Data type for {data_var}",
                "PASS",
                f"Data type '{dtype}' is allowed",
            )
        else:
            report.add(
                SECTION_ID,
                f"Data type for {data_var}",
                "FAIL",
                f"Data type '{dtype}' is not allowed. Allowed types: {allowed_dtypes}",
            )

    return report
