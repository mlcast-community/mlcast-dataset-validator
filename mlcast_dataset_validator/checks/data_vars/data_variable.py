from typing import Dict, Sequence

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from ..data_vars_filter import iter_data_vars
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.4"


@log_function_call
def check_variable_units(
    ds: xr.Dataset,
    *,
    valid_specs: Dict[str, Dict[str, Sequence[str]]],
) -> ValidationReport:
    """
    Validate the units and names of data variables in the dataset.

    Parameters:
        ds (xr.Dataset): The dataset to validate.
        valid_specs (dict): A dictionary specifying valid variable names and their allowed units.

    Returns:
        ValidationReport: A report containing the results of the data variable validation checks.
    """
    report = ValidationReport()

    # Find all grid_mapping data variables since we don't want to check those
    grid_mapping_vars = set()
    for var in ds.data_vars:
        if "grid_mapping" in ds[var].attrs:
            grid_mapping_vars.add(ds[var].attrs["grid_mapping"])
    data_vars = set(ds.data_vars) - grid_mapping_vars

    for data_var in data_vars:
        data_array = ds[data_var]
        units = data_array.attrs.get("units", "").strip()
        valid = False

        for spec_name, spec in valid_specs.items():
            if data_var in spec["names"] and units in spec["units"]:
                valid = True
                report.add(
                    SECTION_ID,
                    f"Validation for {data_var}",
                    "PASS",
                    f"Variable '{data_var}' with units '{units}' is valid under '{spec_name}' specification.",
                )
                break

        if not valid:
            report.add(
                SECTION_ID,
                f"Validation for {data_var}",
                "FAIL",
                f"Variable '{data_var}' with units '{units}' is invalid. "
                f"Expected one of: {valid_specs}.",
            )

    return report


@log_function_call
def check_variable_attributes(
    ds: xr.Dataset,
    *,
    required: Sequence[str] = ("units",),
    recommended: Sequence[str] = ("long_name",),
) -> ValidationReport:
    """
    Check that every data variable carries the required (FAIL if missing) and
    recommended (WARNING if missing) attributes.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.
    required : Sequence[str], optional
        Attributes every data variable MUST carry.
    recommended : Sequence[str], optional
        Attributes every data variable SHOULD carry.

    Returns
    -------
    ValidationReport
        Report with one result per data variable and attribute.
    """
    report = ValidationReport()

    for name, data_array in iter_data_vars(ds):
        for attr, missing_status in [
            *((attr, "FAIL") for attr in required),
            *((attr, "WARNING") for attr in recommended),
        ]:
            requirement = f"Attribute '{attr}' on '{name}'"
            if attr in data_array.attrs:
                report.add(
                    SECTION_ID,
                    requirement,
                    "PASS",
                    f"'{attr}' = '{data_array.attrs[attr]}'",
                )
            else:
                report.add(
                    SECTION_ID,
                    requirement,
                    missing_status,
                    f"Attribute '{attr}' is missing",
                )

    return report
