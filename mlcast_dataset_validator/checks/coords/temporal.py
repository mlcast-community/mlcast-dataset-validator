import weakref
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.3"

_TIMESTEP_CACHE: "weakref.WeakKeyDictionary[xr.Dataset, Tuple[bool, int]]" = (
    weakref.WeakKeyDictionary()
)  # Cache by dataset object to avoid id reuse collisions across tests.


@log_function_call
def check_temporal_requirements(
    ds: xr.Dataset,
    min_years: int,
    allow_variable_timestep: bool,
) -> ValidationReport:
    """
    Validate temporal requirements for the dataset.

    Parameters:
        ds (xr.Dataset): The dataset to validate.
        min_years (int): Minimum required years of continuous temporal coverage.
        allow_variable_timestep (bool): Whether variable timesteps are allowed.

    Returns:
        ValidationReport: A report containing the results of the temporal validation checks.
    """
    report = ValidationReport()

    if "time" not in ds.coords:
        report.add(
            SECTION_ID, "Time coordinate presence", "FAIL", "Missing 'time' coordinate"
        )
        return report

    try:
        time_coord = pd.to_datetime(ds.time.values)
        time_range = time_coord[-1] - time_coord[0]
        years = time_range.days / 365.25
        if years >= min_years:
            report.add(
                SECTION_ID,
                "Minimum 3-year coverage",
                "PASS",
                f"Temporal coverage: {years:.1f} years (≥{min_years} years)",
            )
        else:
            report.add(
                SECTION_ID,
                "Minimum 3-year coverage",
                "FAIL",
                f"Temporal coverage: {years:.1f} years (<{min_years} years required)",
            )
    except Exception as e:
        report.add(
            SECTION_ID,
            "Temporal coverage analysis",
            "FAIL",
            f"Failed to analyze temporal coverage: {e}",
        )
        return report

    report += _check_missing_times_metadata(ds, ds["time"], allow_variable_timestep)
    return report


def _check_missing_times_metadata(
    ds: xr.Dataset,
    time_coord: xr.DataArray,
    allow_variable_timestep: bool,
) -> ValidationReport:
    report = ValidationReport()

    try:
        has_variable_timestep, _ = analyze_dataset_timesteps(ds)
    except Exception as e:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "FAIL",
            f"Failed to analyze timestep variability: {e}",
        )
        return report

    if not has_variable_timestep and "missing_times" not in ds.variables:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "PASS",
            "Regular timestep detected; 'missing_times' not required",
        )
        return report

    if has_variable_timestep and "missing_times" not in ds.variables:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "FAIL",
            "Missing 'missing_times' variable",
        )
        return report

    missing_times_values = ds["missing_times"].values
    if not np.issubdtype(missing_times_values.dtype, np.datetime64):
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "FAIL",
            "'missing_times' must contain datetime64 values",
        )
        return report

    time_values = time_coord.values
    missing_values = missing_times_values.astype("datetime64[ns]")

    overlap = np.intersect1d(time_values, missing_values)
    if overlap.size:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "FAIL",
            "'missing_times' values must not appear in the main time coordinate",
        )
        return report

    consistent_start_raw = ds.attrs.get("consistent_timestep_start")
    regular_start = None
    if allow_variable_timestep and consistent_start_raw is not None:
        try:
            regular_start = pd.to_datetime(consistent_start_raw)
        except Exception as e:
            report.add(
                SECTION_ID,
                "Missing times metadata",
                "FAIL",
                f"Failed to parse 'consistent_timestep_start': {e}",
            )
            return report

    if regular_start is not None:
        pre_times = time_coord.where(time_coord < regular_start, drop=True)
        if pre_times.size >= 2:
            pre_diffs = pre_times.diff("time")
            pre_diffs = pre_diffs.where(pre_diffs > np.timedelta64(0, "ns"), drop=True)
            if pre_diffs.size:
                pre_interval = pre_diffs.min().item()
                expected_pre = pd.date_range(
                    start=pre_times.values[0],
                    end=pre_times.values[-1],
                    freq=pd.to_timedelta(pre_interval),
                ).values
                missing_pre = np.setdiff1d(expected_pre, pre_times.values)
                if missing_pre.size:
                    missing_pre_range = missing_values[
                        (missing_values >= expected_pre[0])
                        & (missing_values <= expected_pre[-1])
                    ]
                    if missing_pre_range.size == 0:
                        report.add(
                            SECTION_ID,
                            "Missing times metadata",
                            "WARNING",
                            "Missing timestamps inferred before consistent_timestep_start are not listed in 'missing_times'",
                        )

    if regular_start is None:
        regular_times = time_coord
    else:
        regular_times = time_coord.where(time_coord >= regular_start, drop=True)

    if regular_times.size < 2:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "WARNING",
            "Not enough regular timestamps to infer missing times",
        )
        return report

    diffs = regular_times.diff("time")
    diffs = diffs.where(diffs > np.timedelta64(0, "ns"), drop=True)
    if diffs.size == 0:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "WARNING",
            "Unable to infer timestep from regular timestamps",
        )
        return report

    expected_interval = diffs.min().item()
    expected_values = pd.date_range(
        start=regular_times.values[0],
        end=regular_times.values[-1],
        freq=pd.to_timedelta(expected_interval),
    ).values
    missing_inferred = np.setdiff1d(expected_values, regular_times.values)

    if missing_inferred.size == 0:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "PASS",
            "No missing timestamps inferred in the regular period",
        )
        return report

    missing_inferred_set = set(missing_inferred.tolist())
    missing_values_set = set(missing_values.tolist())
    missing_unlisted = missing_inferred_set - missing_values_set

    if missing_unlisted:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "FAIL",
            f"{len(missing_unlisted)} inferred missing timestamps are not listed in 'missing_times'",
        )
    else:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "PASS",
            "All inferred missing timestamps are listed in 'missing_times'",
        )

    return report


def analyze_dataset_timesteps(ds: xr.Dataset) -> tuple[bool, int]:
    """
    Analyze the dataset's time coordinate spacing.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing a `time` coordinate whose spacing should be inspected.

    Returns
    -------
    tuple[bool, int]
        Tuple where the first element indicates whether variable intervals exist
        and the second reports how many unique timestep differences were observed.

    Notes
    -----
    Results are cached using the dataset object's Python id to avoid redundant
    computations when the same dataset instance is analyzed multiple times.
    """
    try:
        cached = _TIMESTEP_CACHE.get(ds)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    time_coord = ds["time"]
    if time_coord.size < 2:
        result = (False, 0)
    else:
        diffs = time_coord.diff("time")
        diffs = diffs.where(diffs > np.timedelta64(0, "ns"), drop=True)
        unique_diff_count = len(np.unique(diffs.values))
        result = (unique_diff_count > 1, unique_diff_count)

    try:
        _TIMESTEP_CACHE[ds] = result
    except TypeError:
        pass
    return result
