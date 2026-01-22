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

    da_time_coord = ds["time"]
    if da_time_coord.size >= 2:
        da_time_diffs = da_time_coord.diff("time")
        if not np.all(da_time_diffs.values > np.timedelta64(0, "ns")):
            report.add(
                SECTION_ID,
                "Time coordinate ordering",
                "FAIL",
                "Time coordinate must be strictly increasing",
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
    da_time_coord: xr.DataArray,
    allow_variable_timestep: bool,
) -> ValidationReport:
    """
    Validate missing-times metadata against inferred gaps in the time series.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing time and optional missing_times metadata.
    da_time_coord : xr.DataArray
        Time coordinate values from the dataset.
    allow_variable_timestep : bool
        Whether variable timesteps are allowed by the specification.

    Returns
    -------
    ValidationReport
        Report describing missing_times validation outcomes.
    """
    report = ValidationReport()

    consistent_start_raw = ds.attrs.get("consistent_timestep_start")
    t_regular_start = None
    if allow_variable_timestep and consistent_start_raw is not None:
        try:
            t_regular_start = pd.to_datetime(consistent_start_raw)
        except Exception as e:
            report.add(
                SECTION_ID,
                "Missing times metadata",
                "FAIL",
                f"Failed to parse 'consistent_timestep_start': {e}",
            )
            return report

    if t_regular_start is None:
        da_time_irregular = None
        da_time_regular = da_time_coord
    else:
        irregular_end = t_regular_start - np.timedelta64(1, "ns")
        da_time_irregular = da_time_coord.sel(time=slice(None, irregular_end))
        da_time_regular = da_time_coord.sel(time=slice(t_regular_start, None))

    missing_times_present = "missing_times" in ds.variables
    if not missing_times_present:
        # no missing_times variable present
        missing_values_irregular = None
        missing_values_regular = None
    else:
        # split the missing_times variable into the two time periods, where
        # irregular and regular timesteps are expected
        missing_times_values = ds["missing_times"].values
        if not np.issubdtype(missing_times_values.dtype, np.datetime64):
            report.add(
                SECTION_ID,
                "Missing times metadata",
                "FAIL",
                "'missing_times' must contain datetime64 values",
            )
            return report

        time_values = da_time_coord.values.astype("datetime64[ns]")
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

        if t_regular_start is None:
            # all missing_times pertain to regular period
            missing_values_irregular = None
            missing_values_regular = missing_values
        else:
            # split missing_times into irregular and regular periods
            missing_values_irregular = missing_values[missing_values < t_regular_start]
            missing_values_regular = missing_values[missing_values >= t_regular_start]

    if da_time_irregular is not None:
        report += _check_missing_times_irregular_period(
            da_time_irregular, missing_values_irregular
        )
    report += _check_missing_times_regular_period(
        da_time_regular, missing_values_regular
    )

    if not missing_times_present and not report.has_fails():
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "PASS",
            "Regular timesteps detected; 'missing_times' not required",
        )

    return report


def _check_missing_times_irregular_period(
    da_time_coord: xr.DataArray | None,
    missing_values: np.ndarray | None,
) -> ValidationReport:
    """
    Check irregular timesteps before the regular-timestep period.

    Parameters
    ----------
    da_time_coord : xr.DataArray | None
        Time coordinate values before regular timesteps, if any.
    missing_values : np.ndarray | None
        Array of missing_times values as datetime64, if provided.
    Returns
    -------
    ValidationReport
        Report containing warnings when irregular timesteps do not monotonically
        decrease in size and missing_times do not resolve the inconsistency.
    """
    report = ValidationReport()

    if da_time_coord is None or da_time_coord.size < 2:
        return report

    da_pre_diffs = da_time_coord.diff("time")
    if da_pre_diffs.size < 2:
        return report

    pre_diff_values = da_pre_diffs.values
    is_monotonic_decreasing = np.all(pre_diff_values[1:] <= pre_diff_values[:-1])
    if is_monotonic_decreasing:
        # as long as the irregular timesteps decrease monotonically, no warning is needed
        # we expect the time resolution to decrease in time until the regular period begins
        return report

    if missing_values is None:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "FAIL",
            "Missing 'missing_times' variable for irregular period",
        )
        return report

    pre_interval = da_pre_diffs.min().item()
    da_time_values = da_time_coord.values.astype("datetime64[ns]")
    expected_pre = pd.date_range(
        start=da_time_values[0],
        end=da_time_values[-1],
        freq=pd.to_timedelta(pre_interval),
    ).values
    missing_pre = np.setdiff1d(expected_pre, da_time_values)
    missing_pre_range = missing_values[
        (missing_values >= expected_pre[0]) & (missing_values <= expected_pre[-1])
    ]
    if missing_pre.size and missing_pre_range.size == 0:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "WARNING",
            "Irregular timesteps do not monotonically decrease and missing_times are not provided to resolve them",
        )

    return report


def _check_missing_times_regular_period(
    da_time_coord: xr.DataArray,
    missing_values: np.ndarray | None,
) -> ValidationReport:
    """
    Check inferred missing timestamps within the regular-timestep period.

    Parameters
    ----------
    da_time_coord : xr.DataArray
        Time coordinate values for the regular-timestep period.
    missing_values : np.ndarray | None
        Array of missing_times values as datetime64, if provided.
    Returns
    -------
    ValidationReport
        Report containing pass/fail outcomes for missing_times coverage.
    """
    report = ValidationReport()

    if da_time_coord.size < 2:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "WARNING",
            "Not enough regular timestamps to infer missing times",
        )
        return report

    da_diffs = da_time_coord.diff("time")
    if da_diffs.size == 0:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "WARNING",
            "Unable to infer timestep from regular timestamps",
        )
        return report

    if missing_values is None:
        if len(np.unique(da_diffs.values)) > 1:
            report.add(
                SECTION_ID,
                "Missing times metadata",
                "FAIL",
                "Missing 'missing_times' variable for regular period",
            )
        return report

    expected_interval = da_diffs.min().item()
    da_time_values = da_time_coord.values.astype("datetime64[ns]")
    expected_values = pd.date_range(
        start=da_time_values[0],
        end=da_time_values[-1],
        freq=pd.to_timedelta(expected_interval),
    ).values
    missing_inferred = np.setdiff1d(expected_values, da_time_values)

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
        missing_list = sorted(missing_unlisted)
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "FAIL",
            "Inferred missing timestamps are not listed in 'missing_times': "
            f"{missing_list}. Define these in 'missing_times' if the data is actually missing.",
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
    Results are cached using the dataset object to avoid redundant computations
    when the same dataset instance is analyzed multiple times.
    """
    try:
        cached = _TIMESTEP_CACHE.get(ds)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    da_time_coord = ds["time"]
    if da_time_coord.size < 2:
        result = (False, 0)
    else:
        da_diffs = da_time_coord.diff("time")
        da_diffs = da_diffs.where(da_diffs > np.timedelta64(0, "ns"), drop=True)
        unique_diff_count = len(np.unique(da_diffs.values))
        result = (unique_diff_count > 1, unique_diff_count)

    try:
        _TIMESTEP_CACHE[ds] = result
    except TypeError:
        pass
    return result
