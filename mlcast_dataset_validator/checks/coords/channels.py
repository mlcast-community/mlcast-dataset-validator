from __future__ import annotations

from typing import List

import numpy as np
import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.4"

# Units accepted for the central wavelength of a spectral channel
_WAVELENGTH_UNITS = {"m", "um", "µm", "micrometer", "micrometre", "micron", "nm"}


def _find_channel_coords(ds: xr.Dataset, channel_dim: str) -> List[str]:
    """
    Find the 1-D coordinates that lie along the channel dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to search.
    channel_dim : str
        Name of the channel dimension.

    Returns
    -------
    List[str]
        Names of the coordinates with dims ``(channel_dim,)``.
    """
    return [name for name, da in ds.coords.items() if da.dims == (channel_dim,)]


@log_function_call
def check_channel_coordinate(
    ds: xr.Dataset, *, channel_dim: str = "channel"
) -> ValidationReport:
    """
    Validate the labelling of the spectral channel dimension.

    The channel dimension must have a 1-D coordinate of unique string labels
    (e.g. ``"ir_105"``), so that a channel can be selected by name with
    ``ds.sel(channel=...)``. Each channel should also have a central wavelength
    coordinate with ``standard_name: radiation_wavelength`` and a length unit.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.
    channel_dim : str, optional
        Name of the channel dimension, by default ``"channel"``.

    Returns
    -------
    ValidationReport
        A report with the results of the channel coordinate checks.
    """
    report = ValidationReport()

    if channel_dim not in ds.dims:
        report.add(
            SECTION_ID,
            "Channel dimension",
            "FAIL",
            f"Dataset has no '{channel_dim}' dimension for the spectral channels",
        )
        return report

    channel_coords = _find_channel_coords(ds, channel_dim)
    label_coords = [
        name for name in channel_coords if ds[name].dtype.kind in ("U", "O", "S")
    ]
    if not label_coords:
        hint = ""
        candidates = [
            name
            for name, da in ds.data_vars.items()
            if da.dims == (channel_dim,) and da.dtype.kind in ("U", "O", "S")
        ]
        if candidates:
            hint = (
                f" Data variable(s) {candidates} hold string values along '{channel_dim}';"
                f" store the labels as the '{channel_dim}' coordinate instead"
                f" (e.g. `ds.set_coords({candidates[0]!r})` or"
                f" `ds.assign_coords({channel_dim}=ds[{candidates[0]!r}])`)."
            )
        report.add(
            SECTION_ID,
            "Channel labels",
            "FAIL",
            f"The '{channel_dim}' dimension has no coordinate with channel name labels."
            + hint,
        )
    else:
        name = channel_dim if channel_dim in label_coords else label_coords[0]
        values = ds[name].values
        if ds[name].dtype.kind == "S":
            report.add(
                SECTION_ID,
                "Channel labels",
                "FAIL",
                f"Channel labels in '{name}' are bytes ({ds[name].dtype}), not strings."
                " Store them as strings, or as fixed-width bytes with the attribute"
                " `_Encoding: utf-8` so that xarray decodes them to strings.",
            )
        elif len(np.unique(values.astype(str))) != len(values):
            report.add(
                SECTION_ID,
                "Channel labels",
                "FAIL",
                f"Channel labels in '{name}' are not unique: {list(values)}",
            )
        else:
            report.add(
                SECTION_ID,
                "Channel labels",
                "PASS",
                f"Channels are labelled by '{name}': {list(values)}",
            )

    wavelength_coords = [
        name
        for name in channel_coords
        if ds[name].attrs.get("standard_name") == "radiation_wavelength"
    ]
    if not wavelength_coords:
        report.add(
            SECTION_ID,
            "Channel central wavelength",
            "WARNING",
            f"No coordinate along '{channel_dim}' with standard_name"
            " 'radiation_wavelength' giving the central wavelength of each channel",
        )
    else:
        name = wavelength_coords[0]
        units = str(ds[name].attrs.get("units", "")).strip()
        if units.lower() in _WAVELENGTH_UNITS:
            report.add(
                SECTION_ID,
                "Channel central wavelength",
                "PASS",
                f"Central wavelengths are given in '{name}' ({units})",
            )
        else:
            report.add(
                SECTION_ID,
                "Channel central wavelength",
                "WARNING",
                f"Central wavelength coordinate '{name}' has units '{units}', expected"
                f" a length unit (one of {sorted(_WAVELENGTH_UNITS)})",
            )

    return report
