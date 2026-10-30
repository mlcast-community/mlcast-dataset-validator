from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from ..data_vars_filter import classify_data_vars
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.6"


@log_function_call
def check_counts_calibration(
    ds: xr.Dataset,
    *,
    spatial_dims: Sequence[str] = ("y", "x"),
    calibration_dims: Sequence[str] = ("time", "channel"),
) -> ValidationReport:
    """
    Validate that raw detector counts are linked to their calibration coefficients.

    Every observation variable (see
    :func:`~mlcast_dataset_validator.checks.data_vars_filter.classify_data_vars`)
    must:

    - be stored on disk as integers (``encoding["dtype"]``) and have
      ``units: "1"``,
    - list its calibration coefficient variables in ``ancillary_variables``.
      These must exist, must not span the spatial dimensions, may only use
      ``calibration_dims`` and must carry ``long_name`` and ``units``,
    - preferably have a ``comment`` describing how to convert the counts with
      the coefficients (WARNING if missing).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.
    spatial_dims : Sequence[str], optional
        Names of the spatial dimensions, by default ``("y", "x")``.
    calibration_dims : Sequence[str], optional
        Dimensions the calibration coefficients may vary along, by default
        ``("time", "channel")``.

    Returns
    -------
    ValidationReport
        A report with the results of the counts calibration checks.
    """
    report = ValidationReport()
    observations = classify_data_vars(ds, spatial_dims=spatial_dims)["observation"]
    if not observations:
        report.add(
            SECTION_ID,
            "Counts variables",
            "FAIL",
            f"No data variable spanning the spatial dims {tuple(spatial_dims)} found",
        )
        return report

    for var_name in observations:
        da = ds[var_name]

        on_disk_dtype = np.dtype(da.encoding.get("dtype", da.dtype))
        units = str(da.attrs.get("units", "")).strip()
        problems = []
        if on_disk_dtype.kind not in ("i", "u"):
            problems.append(f"stored as {on_disk_dtype}, not as integers")
        if units != "1":
            problems.append(f"has units '{units}', expected '1'")
        if problems:
            report.add(
                SECTION_ID,
                f"Counts encoding for {var_name}",
                "FAIL",
                f"Counts variable '{var_name}' " + " and ".join(problems),
            )
        else:
            report.add(
                SECTION_ID,
                f"Counts encoding for {var_name}",
                "PASS",
                f"Counts variable '{var_name}' is stored as {on_disk_dtype} with units '1'",
            )

        refs = str(da.attrs.get("ancillary_variables", "")).split()
        coefficient_vars = [
            ref
            for ref in refs
            if ref in ds.variables
            and not any(dim in ds[ref].dims for dim in spatial_dims)
        ]
        requirement = f"Calibration coefficients for {var_name}"
        missing_refs = [ref for ref in refs if ref not in ds.variables]
        if missing_refs:
            report.add(
                SECTION_ID,
                requirement,
                "FAIL",
                f"'ancillary_variables' of '{var_name}' references variables that do not"
                f" exist: {missing_refs}",
            )
        if not coefficient_vars:
            report.add(
                SECTION_ID,
                requirement,
                "FAIL",
                f"'{var_name}' does not reference any calibration coefficient variables"
                " (e.g. slope/offset) in its 'ancillary_variables' attribute"
                f" (currently {refs or 'not set'})",
            )
        for ref in coefficient_vars:
            da_coeff = ds[ref]
            problems = [
                f"is missing '{attr}'"
                for attr in ("long_name", "units")
                if attr not in da_coeff.attrs
            ]
            extra_dims = [d for d in da_coeff.dims if d not in calibration_dims]
            if extra_dims:
                problems.append(
                    f"varies along {extra_dims}, only {tuple(calibration_dims)} are allowed"
                )
            if problems:
                report.add(
                    SECTION_ID,
                    requirement,
                    "FAIL",
                    f"Calibration variable '{ref}' " + " and ".join(problems),
                )
            else:
                report.add(
                    SECTION_ID,
                    requirement,
                    "PASS",
                    f"Calibration variable '{ref}' {da_coeff.dims} has long_name and"
                    f" units '{da_coeff.attrs['units']}'",
                )

        if "comment" in da.attrs:
            report.add(
                SECTION_ID,
                f"Calibration description for {var_name}",
                "PASS",
                f"'{var_name}' describes the calibration: {da.attrs['comment']}",
            )
        else:
            report.add(
                SECTION_ID,
                f"Calibration description for {var_name}",
                "WARNING",
                f"'{var_name}' has no 'comment' attribute describing how to convert the"
                " counts to radiances with the calibration coefficients",
            )

    return report
