from __future__ import annotations

from typing import Sequence

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.5"


@log_function_call
def check_platform_attributes(
    ds: xr.Dataset, *, required: Sequence[str] = ("platform", "instrument")
) -> ValidationReport:
    """
    Validate the global attributes describing the observing platform.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.
    required : Sequence[str], optional
        Global attributes that must be present as non-empty strings, by
        default ``("platform", "instrument")``.

    Returns
    -------
    ValidationReport
        A report with one PASS or FAIL entry per required attribute.
    """
    report = ValidationReport()
    for attr in required:
        value = ds.attrs.get(attr)
        if isinstance(value, str) and value.strip():
            report.add(
                SECTION_ID,
                f"Global attribute '{attr}'",
                "PASS",
                f"'{attr}' is '{value}'",
            )
        else:
            report.add(
                SECTION_ID,
                f"Global attribute '{attr}'",
                "FAIL",
                f"Global attribute '{attr}' is missing or empty",
            )
    return report
