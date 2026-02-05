import numpy as np
import pandas as pd
import xarray as xr

from mlcast_dataset_validator.checks.coords.temporal import check_temporal_requirements


def _make_dataset(time_values, missing_times=None, attrs=None):
    data = np.zeros(len(time_values), dtype="float32")
    coords = {"time": pd.to_datetime(time_values)}
    if missing_times is not None:
        coords["missing_times"] = (
            "missing_times",
            pd.to_datetime(missing_times),
        )
    ds = xr.Dataset({"precip": ("time", data)}, coords=coords)
    if attrs:
        ds.attrs.update(attrs)
    return ds


def test_missing_times_not_required_for_regular_timestep():
    """Regular timesteps pass without missing_times.

    Timeline: 00-01-02-03-04 (all present)
    """
    time_values = [
        "2020-01-01T00:00:00",
        "2020-01-01T01:00:00",
        "2020-01-01T02:00:00",
        "2020-01-01T03:00:00",
        "2020-01-01T04:00:00",
    ]
    ds = _make_dataset(time_values)

    report = check_temporal_requirements(ds, min_years=0, allow_variable_timestep=True)

    assert not report.has_fails()


def test_missing_times_required_for_variable_timestep():
    """
    Variable timesteps fail without missing_times.

    Timeline: 00-01-(02 missing)-03-04
    No missing_times provided.
    """
    time_values = [
        "2020-01-01T00:00:00",
        "2020-01-01T01:00:00",
        "2020-01-01T03:00:00",
        "2020-01-01T04:00:00",
    ]
    ds = _make_dataset(time_values)

    report = check_temporal_requirements(ds, min_years=0, allow_variable_timestep=True)

    assert report.has_fails()


def test_missing_times_fail_when_missing_timestamp_unlisted():
    """
    Fails when inferred missing timestamp is not listed in missing_times.

    Timeline: 00-01-(02 missing)-03-04-(05 missing)-06, missing_times=[]
    """
    time_values = [
        "2020-01-01T00:00:00",
        "2020-01-01T01:00:00",
        "2020-01-01T03:00:00",
        "2020-01-01T04:00:00",
        "2020-01-01T06:00:00",
    ]
    missing_times = []
    ds = _make_dataset(
        time_values,
        missing_times=missing_times,
    )

    report = check_temporal_requirements(ds, min_years=0, allow_variable_timestep=True)

    assert report.has_fails()


def test_missing_times_pass_when_missing_timestamp_listed():
    """Passes when inferred missing timestamp is listed in missing_times.

    Timeline: 00-01-(02 missing)-03-04, missing_times=[02]
    """
    time_values = [
        "2020-01-01T00:00:00",
        "2020-01-01T01:00:00",
        "2020-01-01T03:00:00",
        "2020-01-01T04:00:00",
    ]
    missing_times = ["2020-01-01T02:00:00"]
    ds = _make_dataset(
        time_values,
        missing_times=missing_times,
    )

    report = check_temporal_requirements(ds, min_years=0, allow_variable_timestep=True)

    assert not report.has_fails()


def test_time_values_must_be_monotonically_increasing():
    """Fails when time values are not strictly increasing.

    Timeline: 00-02-01 (decreasing step)
    """
    time_values = [
        "2020-01-01T00:00:00",
        "2020-01-01T02:00:00",
        "2020-01-01T01:00:00",
    ]
    ds = _make_dataset(time_values)

    report = check_temporal_requirements(ds, min_years=0, allow_variable_timestep=True)

    assert report.has_fails()


def test_warning_for_missing_times_before_consistent_timestep_start():
    """
    Warns when missing timestamps appear before consistent_timestep_start with no missing_times in that range.

    Timeline: 00-01-(02 missing)-03 | 04-05 (regular)
              ^ consistent_timestep_start at 04
    """
    time_values = [
        "2020-01-01T00:00:00",
        "2020-01-01T01:00:00",
        "2020-01-01T03:00:00",
        "2020-01-01T04:00:00",
        "2020-01-01T05:00:00",
    ]
    ds = _make_dataset(
        time_values,
        missing_times=[],
        attrs={"consistent_timestep_start": "2020-01-01T04:00:00"},
    )

    report = check_temporal_requirements(ds, min_years=0, allow_variable_timestep=True)

    assert report.has_warnings()


def test_pass_for_missing_times_before_consistent_timestep_start():
    """
    With missing_times listed before consistent_timestep_start no warnings should be issued.

    Timeline: 00-02-04-05-(06 missing)-07 | 08-09-10 (regular)
    """
    time_values = [
        "2020-01-01T00:00:00",
        "2020-01-01T02:00:00",  # 2hr timestep
        "2020-01-01T04:00:00",  # 2hr timestep
        "2020-01-01T05:00:00",  # 1hr timestep
        "2020-01-01T07:00:00",  # should be 06:00 missing for 1hr timestep, but 06:00 is missing
        "2020-01-01T08:00:00",  # 1hr, regular timesteps begin
        "2020-01-01T09:00:00",
        "2020-01-01T10:00:00",
    ]
    ds = _make_dataset(
        time_values,
        missing_times=["2020-01-01T06:00:00"],
        attrs={"consistent_timestep_start": "2020-01-01T08:00:00"},
    )

    report = check_temporal_requirements(ds, min_years=0, allow_variable_timestep=True)

    assert not report.has_fails()
    assert not report.has_warnings()


def test_pass_monotonic_variable_timestep_before_consistent_timestep_start():
    """
    With missing_times listed before consistent_timestep_start no warnings should be issued.

    Timeline: 00-04-08-10 | 11-12-13 (regular)
    """
    time_values = [
        "2020-01-01T00:00:00",
        "2020-01-01T04:00:00",  # 4hr timestep
        "2020-01-01T08:00:00",  # 4hr timestep
        "2020-01-01T10:00:00",  # 2hr timestep
        "2020-01-01T11:00:00",  # 1hr timestep (regular timesteps begin)
        "2020-01-01T12:00:00",  # 1hr timestep
        "2020-01-01T13:00:00",  # 1hr timestep
    ]
    ds = _make_dataset(
        time_values,
        attrs={"consistent_timestep_start": "2020-01-01T11:00:00"},
    )

    report = check_temporal_requirements(ds, min_years=0, allow_variable_timestep=True)

    assert not report.has_fails()
    assert not report.has_warnings()


def test_variable_timestep_not_allowed_fails():
    """Variable timesteps fail when not allowed.

    Timeline: 00-01-(02 missing)-03-04
    """
    time_values = [
        "2020-01-01T00:00:00",
        "2020-01-01T01:00:00",
        "2020-01-01T03:00:00",
        "2020-01-01T04:00:00",
    ]
    ds = _make_dataset(time_values)

    report = check_temporal_requirements(ds, min_years=0, allow_variable_timestep=False)

    assert report.has_fails()
