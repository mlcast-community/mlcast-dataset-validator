from __future__ import annotations

from typing import Dict, Iterable

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from ..data_vars_filter import iter_data_vars
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.4"

# _CF_RULES maps a CF `standard_name` to the acceptable variable names,
# permissible units, and the canonical CF unit for that physical quantity.
_CF_RULES: Dict[str, Dict[str, Iterable[str]]] = {
    "rainfall_flux": {
        "names": ("mmh", "rr", "rain_rate", "rainfall_rate", "rainfall_flux"),
        "units": ("kg m-2 h-1", "mm h-1", "mm/h"),
        "canonical_unit": "kg m-2 h-1",
    },
    "precipitation_flux": {
        "names": ("tprate", "prate"),
        "units": ("kg m-2 h-1", "mm h-1", "mm/h"),
        "canonical_unit": "kg m-2 h-1",
    },
    "equivalent_reflectivity_factor": {
        "names": ("equivalent_reflectivity_factor", "dbz", "rare"),
        "units": ("dBZ",),
        "canonical_unit": "dBZ",
    },
    "precipitation_amount": {
        "names": ("rainfall_amount", "mm", "precipitation_amount", "tp"),
        "units": ("kg m-2", "mm"),
        "canonical_unit": "kg m-2",
    },
    "rainfall_amount": {
        "names": ("rainfall_amount", "mm", "precipitation_amount", "tp"),
        "units": ("kg m-2", "mm"),
        "canonical_unit": "kg m-2",
    },
}


@log_function_call
def check_names_and_attrs(
    ds: xr.Dataset, *, allowed_standard_names: Iterable[str]
) -> ValidationReport:
    """
    Validate that data variables match the expected CF naming/attribute rules.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose data variables should be evaluated.
    allowed_standard_names : Iterable[str]
        CF `standard_name` values that this spec considers valid. Every data
        variable encountered must declare a standard name from this list; any
        other names will trigger failures.

    Returns
    -------
    ValidationReport
        A report containing PASS/FAIL/WARNING entries for each relevant variable.
    """
    report = ValidationReport()

    allowed_standard_names = {name.lower() for name in allowed_standard_names}
    for spec_name in allowed_standard_names:
        if spec_name not in _CF_RULES:
            raise NotImplementedError(
                f"No naming rule implemented for standard_name '{spec_name}'"
            )

    for var_name, data_array in iter_data_vars(ds):
        missing_attr = next(
            (
                attr
                for attr in ("long_name", "standard_name", "units")
                if attr not in data_array.attrs
            ),
            None,
        )
        if missing_attr:
            report.add(
                SECTION_ID,
                f"Required attribute '{missing_attr}'",
                "FAIL",
                f"'{missing_attr}' attribute is missing on data variable '{var_name}'.",
            )
            continue

        standard_name = data_array.attrs.get("standard_name", "").strip().lower()
        units = data_array.attrs.get("units", "").strip()
        canonical_name = var_name.strip().lower()

        if standard_name not in allowed_standard_names:
            report.add(
                SECTION_ID,
                "Standard name validation",
                "FAIL",
                f"Standard name '{standard_name}' is not permitted for this specification.",
            )
            continue

        matched_rule = _CF_RULES.get(standard_name)
        if matched_rule is None:
            report.add(
                SECTION_ID,
                "Standard name validation",
                "FAIL",
                f"Standard name '{standard_name}' is not recognized for supported physical variables.",
            )
            continue

        allowed_names = {name.lower() for name in matched_rule["names"]}
        if canonical_name not in allowed_names:
            report.add(
                SECTION_ID,
                "Variable name validation",
                "FAIL",
                f"Variable name '{var_name}' is not allowed for standard_name '{standard_name}'. "
                f"Allowed names: {', '.join(matched_rule['names'])}.",
            )
        else:
            report.add(
                SECTION_ID,
                "Variable name validation",
                "PASS",
                f"Variable name '{var_name}' matches the expected CF/ECMWF list for standard_name '{standard_name}'.",
            )

        allowed_units = set(matched_rule["units"])
        canonical_unit = matched_rule["canonical_unit"]
        if units not in allowed_units:
            report.add(
                SECTION_ID,
                "Units validation",
                "FAIL",
                f"Units '{units}' are not allowed for standard_name '{standard_name}'. "
                f"Allowed units: {', '.join(allowed_units)} (canonical: {canonical_unit}).",
            )
        elif units != canonical_unit:
            report.add(
                SECTION_ID,
                "Units validation",
                "WARNING",
                f"Units '{units}' are permitted but the CF canonical unit is '{canonical_unit}'.",
            )
        else:
            report.add(
                SECTION_ID,
                "Units validation",
                "PASS",
                f"Units '{units}' match the CF canonical unit.",
            )

    return report
