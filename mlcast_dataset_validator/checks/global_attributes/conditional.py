from typing import Sequence

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from ..coords.temporal import analyze_dataset_timesteps
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.1"


def _requires_variable_timestep_metadata(ds: xr.Dataset) -> bool:
    """Return True if the dataset exhibits variable timesteps."""
    if "time" not in ds.coords:
        return False

    has_variable, _ = analyze_dataset_timesteps(ds)
    return has_variable


CONDITION_FUNCTIONS = dict(
    consistent_timestep_start=_requires_variable_timestep_metadata,
)


@log_function_call
def check_conditional_global_attributes(
    ds: xr.Dataset,
    *,
    conditional_attrs: Sequence[str] = (),
) -> ValidationReport:
    """
    Check global attributes.

    The following conditional global attributes are implemented:

    - consistent_timestep_start: An ISO 8601 timestamp MAY be present if the dataset has variable timestepping
    """

    report = ValidationReport()

    # Check conditional attributes
    for attr in conditional_attrs:
        condition_fn = CONDITION_FUNCTIONS.get(attr)
        if condition_fn is None:
            raise NotImplementedError(
                f"Conditional attribute check for '{attr}' is not implemented."
            )
        is_required = condition_fn(ds)
        if is_required:
            if attr in ds.attrs:
                report.add(
                    SECTION_ID,
                    f"Conditional global attribute '{attr}'",
                    "PASS",
                    f"Global attribute '{attr}' is present",
                )
            else:
                report.add(
                    SECTION_ID,
                    f"Conditional global attribute '{attr}'",
                    "FAIL",
                    f"Global attribute '{attr}' is required but not present",
                )

    return report
