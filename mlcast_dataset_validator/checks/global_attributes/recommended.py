from typing import Sequence

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.5"


@log_function_call
def check_global_attributes(
    ds: xr.Dataset,
    *,
    required: Sequence[str] = (),
    recommended: Sequence[str] = (),
) -> ValidationReport:
    """
    Check that required (FAIL if missing) and recommended (WARNING if missing)
    global attributes are present and non-empty.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.
    required : Sequence[str], optional
        Global attributes that MUST be present.
    recommended : Sequence[str], optional
        Global attributes that SHOULD be present.

    Returns
    -------
    ValidationReport
        Report with one result per attribute.
    """
    report = ValidationReport()

    for attr, missing_status in [
        *((attr, "FAIL") for attr in required),
        *((attr, "WARNING") for attr in recommended),
    ]:
        value = ds.attrs.get(attr)
        if value is None or str(value).strip() == "":
            report.add(
                SECTION_ID,
                f"Global attribute '{attr}'",
                missing_status,
                f"Global attribute '{attr}' is missing or empty",
            )
        else:
            report.add(
                SECTION_ID,
                f"Global attribute '{attr}'",
                "PASS",
                f"Global attribute '{attr}' is present",
            )

    return report
