from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.4"

# Each kind maps to the numpy dtype kind characters it accepts.
_DTYPE_KINDS: Dict[str, str] = {
    "string": "USO",
    "numeric": "iuf",
    "datetime": "M",
    "timedelta": "m",
}


def _is_kind(values: np.ndarray, kind: str) -> bool:
    """Return True if `values` has a dtype of the given kind."""
    if values.dtype.kind != "O":
        return values.dtype.kind in _DTYPE_KINDS[kind]
    # object arrays are only accepted as strings, and only if every value is one
    return kind == "string" and all(isinstance(v, str) for v in values.ravel())


@log_function_call
def check_required_dimensions(
    ds: xr.Dataset,
    *,
    required: Sequence[str],
    forbidden: Sequence[str] = (),
) -> ValidationReport:
    """
    Check that required dimensions exist, forbidden ones don't, and that every
    dimension has a coordinate variable.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.
    required : Sequence[str]
        Dimension names that MUST be present.
    forbidden : Sequence[str], optional
        Dimension names that MUST NOT be present.

    Returns
    -------
    ValidationReport
        Report with one result per required/forbidden dimension, and a FAIL
        for every dimension without a coordinate variable.
    """
    report = ValidationReport()

    for dim in required:
        if dim in ds.dims:
            report.add(
                SECTION_ID,
                f"Required dimension '{dim}'",
                "PASS",
                f"Dimension '{dim}' is present",
            )
        else:
            report.add(
                SECTION_ID,
                f"Required dimension '{dim}'",
                "FAIL",
                f"Dimension '{dim}' is missing (found {list(ds.dims)})",
            )

    for dim in forbidden:
        if dim in ds.dims:
            report.add(
                SECTION_ID,
                f"Forbidden dimension '{dim}'",
                "FAIL",
                f"Dimension '{dim}' MUST NOT be present",
            )
        else:
            report.add(
                SECTION_ID,
                f"Forbidden dimension '{dim}'",
                "PASS",
                f"Dimension '{dim}' is absent",
            )

    missing_coords = [dim for dim in ds.dims if dim not in ds.coords]
    if missing_coords:
        report.add(
            SECTION_ID,
            "Dimension coordinates",
            "FAIL",
            f"Dimension(s) {missing_coords} have no coordinate variable",
        )
    else:
        report.add(
            SECTION_ID,
            "Dimension coordinates",
            "PASS",
            "Every dimension has a coordinate variable",
        )

    return report


@log_function_call
def check_dimension_coordinate_types(
    ds: xr.Dataset,
    *,
    kinds: Dict[str, str],
    recommended_attrs: Dict[str, Sequence[str]] | None = None,
) -> ValidationReport:
    """
    Check the dtype and uniqueness of dimension coordinates.

    Dimensions listed in `kinds` but absent from the dataset are skipped, use
    `check_required_dimensions` to require their presence.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.
    kinds : Dict[str, str]
        Mapping of dimension name to the kind of values its coordinate MUST
        hold, one of "string", "numeric", "datetime" or "timedelta".
    recommended_attrs : Dict[str, Sequence[str]], optional
        Mapping of dimension name to attributes its coordinate SHOULD carry.

    Returns
    -------
    ValidationReport
        Report with dtype and uniqueness results for every present dimension.
    """
    report = ValidationReport()
    recommended_attrs = recommended_attrs or {}

    for dim, kind in kinds.items():
        if kind not in _DTYPE_KINDS:
            raise ValueError(
                f"Unknown kind '{kind}' for '{dim}', expected one of {list(_DTYPE_KINDS)}"
            )
        if dim not in ds.dims or dim not in ds.coords:
            continue

        values = ds[dim].values
        if _is_kind(values, kind):
            report.add(
                SECTION_ID,
                f"Coordinate '{dim}' type",
                "PASS",
                f"Coordinate '{dim}' holds {kind} values ({values.dtype})",
            )
        else:
            report.add(
                SECTION_ID,
                f"Coordinate '{dim}' type",
                "FAIL",
                f"Coordinate '{dim}' MUST hold {kind} values, found dtype {values.dtype}",
            )

        n_unique = len(np.unique(values))
        if n_unique == values.size:
            report.add(
                SECTION_ID,
                f"Coordinate '{dim}' uniqueness",
                "PASS",
                f"All {values.size} values are unique",
            )
        else:
            report.add(
                SECTION_ID,
                f"Coordinate '{dim}' uniqueness",
                "FAIL",
                f"{values.size - n_unique} duplicate value(s) found",
            )

    for dim, attrs in recommended_attrs.items():
        if dim not in ds.coords:
            continue
        missing = [attr for attr in attrs if attr not in ds[dim].attrs]
        if missing:
            report.add(
                SECTION_ID,
                f"Coordinate '{dim}' attributes",
                "WARNING",
                f"Coordinate '{dim}' SHOULD carry attribute(s) {missing}",
            )
        else:
            report.add(
                SECTION_ID,
                f"Coordinate '{dim}' attributes",
                "PASS",
                f"Coordinate '{dim}' carries {list(attrs)}",
            )

    return report
