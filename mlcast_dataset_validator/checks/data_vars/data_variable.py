from typing import Dict, Sequence

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
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
