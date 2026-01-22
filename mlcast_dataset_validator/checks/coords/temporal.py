import weakref
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.3"
VARIABLE_TIMESTEP_SECTION_ID = f"{PARENT_SECTION_ID}.4"

_TIMESTEP_CACHE: "weakref.WeakKeyDictionary[xr.Dataset, Tuple[bool, int]]" = (
    weakref.WeakKeyDictionary()
)  # Cache by dataset object to avoid id reuse collisions across tests.


@log_function_call
def check_variable_timestep(
    ds: xr.Dataset,
    *,
    allow_variable_timestep: bool,
) -> ValidationReport:
    """
    Validate whether the dataset's timestep pattern satisfies the specification.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to evaluate.
    allow_variable_timestep : bool
        Whether multiple unique intervals are allowed by the spec variant.

    Returns
    -------
    ValidationReport
        Populated report describing the outcome of each timestep-related rule.
    """
    report = ValidationReport()

    if "time" not in ds.coords:
        report.add(
            VARIABLE_TIMESTEP_SECTION_ID,
            "Time coordinate presence",
            "FAIL",
            "Missing 'time' coordinate",
        )
        return report

    try:
        has_variable, unique_diff_count = analyze_dataset_timesteps(ds)
    except Exception as e:
        report.add(
            VARIABLE_TIMESTEP_SECTION_ID,
            "Variable timestep analysis",
            "FAIL",
            f"Failed to analyze variable timesteps: {e}",
        )
        return report

    if not has_variable:
        report.add(
            VARIABLE_TIMESTEP_SECTION_ID,
            "Timestep consistency",
            "PASS",
            "Timestep is consistent throughout the dataset",
        )
        return report

    if allow_variable_timestep:
        report.add(
            VARIABLE_TIMESTEP_SECTION_ID,
            "Variable timestep handling",
            "PASS",
            f"Variable timesteps detected with {unique_diff_count} unique intervals",
        )
    else:
        report.add(
            VARIABLE_TIMESTEP_SECTION_ID,
            "Variable timestep handling",
            "FAIL",
            "Variable timesteps detected but not allowed",
        )

    if "consistent_timestep_start" in ds.attrs:
        report.add(
            VARIABLE_TIMESTEP_SECTION_ID,
            "Consistent timestep start metadata",
            "PASS",
            "Dataset includes 'consistent_timestep_start' metadata",
        )
    else:
        report.add(
            VARIABLE_TIMESTEP_SECTION_ID,
            "Consistent timestep start metadata",
            "WARNING",
            "Dataset has variable timesteps but is missing 'consistent_timestep_start' metadata",
        )

    return report


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

    try:
        time_values = time_coord.asi8
    except AttributeError:
        time_values = pd.Series(time_coord).astype("int64").to_numpy()

    missing_values = missing_times_values.astype("datetime64[ns]").astype("int64")

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
        pre_mask = time_coord < regular_start
        pre_values = time_values[pre_mask]
        if len(pre_values) >= 2:
            pre_diffs = np.diff(pre_values)
            pre_diffs = pre_diffs[pre_diffs > 0]
            if pre_diffs.size:
                pre_interval = int(pre_diffs.min())
                expected_pre = np.arange(
                    int(pre_values[0]),
                    int(pre_values[-1]) + pre_interval,
                    pre_interval,
                    dtype="int64",
                )
                missing_pre = np.setdiff1d(expected_pre, pre_values)
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
        regular_mask = np.ones(time_values.shape, dtype=bool)
    else:
        regular_mask = time_coord >= regular_start

    regular_values = time_values[regular_mask]
    if len(regular_values) < 2:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "WARNING",
            "Not enough regular timestamps to infer missing times",
        )
        return report

    diffs = np.diff(regular_values)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "WARNING",
            "Unable to infer timestep from regular timestamps",
        )
        return report

    expected_interval = int(diffs.min())
    expected_values = np.arange(
        int(regular_values[0]),
        int(regular_values[-1]) + expected_interval,
        expected_interval,
        dtype="int64",
    )
    missing_inferred = np.setdiff1d(expected_values, regular_values)

    if missing_inferred.size == 0:
        report.add(
            SECTION_ID,
            "Missing times metadata",
            "PASS",
            "No missing timestamps inferred in the regular period",
        )
        return report

    missing_inferred_set = set(int(val) for val in missing_inferred)
    missing_values_set = set(int(val) for val in missing_values)
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

    time_index = pd.to_datetime(ds.time.values)
    try:
        raw_values = time_index.asi8
    except AttributeError:
        raw_values = pd.Series(time_index).astype("int64").to_numpy()

    if len(raw_values) < 2:
        result = (False, 0)
    else:
        unique_diffs = {
            int(raw_values[idx + 1]) - int(raw_values[idx])
            for idx in range(len(raw_values) - 1)
        }
        unique_diff_count = len(unique_diffs)
        result = (unique_diff_count > 1, unique_diff_count)

    try:
        _TIMESTEP_CACHE[ds] = result
    except TypeError:
        pass
    return result
