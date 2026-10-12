import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scripts import make_verification_scores_example as example

from mlcast_dataset_validator.checks.global_attributes import mlcast_metadata
from mlcast_dataset_validator.checks.global_attributes.zarr_format import (
    check_zarr_format,
)
from mlcast_dataset_validator.specs import cli
from mlcast_dataset_validator.specs.reporting import ValidationReport, skip_all_checks
from mlcast_dataset_validator.specs.verification import forecast_scoring_metrics


def make_metrics_dataset(**kwargs):
    """Valid forecast scoring metrics dataset, modelled on the smallest input
    that mlwp-scorecard renders (see its README)."""
    return example.make_verification_scores_example(validator_version="0.4.0", **kwargs)


def _validate(ds):
    report, _ = forecast_scoring_metrics.validate_dataset(ds)
    return report


def _statuses(report, section):
    return {r.status for r in report.results if r.section == section}


@pytest.fixture(autouse=True)
def offline_metadata(monkeypatch):
    monkeypatch.setattr(mlcast_metadata, "__version__", "0.4.0")
    monkeypatch.setattr(mlcast_metadata, "github_repo_exists", lambda *a: True)
    monkeypatch.setattr(mlcast_metadata, "github_ref_exists", lambda *a: "tags/v0.1.0")


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_valid_dataset_passes(tmp_path, suffix):
    path = tmp_path / f"scores{suffix}"
    ds = make_metrics_dataset()
    if suffix == ".nc":
        ds.to_netcdf(path)
        ds = xr.open_dataset(path)
    else:
        ds.to_zarr(path)
        ds = xr.open_zarr(path)

    report = _validate(ds)

    assert not report.has_fails(), [r for r in report.results if r.status == "FAIL"]
    assert not report.has_warnings(), [
        r for r in report.results if r.status == "WARNING"
    ]
    sections = {r.section for r in report.results}
    assert sections == {"3.1", "3.2", "3.3", "4.1", "4.2", "4.3", "4.4"} | {
        "5.1",
        "5.2",
        "5.3",
        "5.4",
    }


def test_example_script_writes_valid_dataset(tmp_path, monkeypatch):
    path = tmp_path / "scores.nc"
    example.main([str(path)])
    monkeypatch.setattr(
        "sys.argv", ["prog", "verification", "forecast_scoring_metrics", str(path)]
    )
    assert cli.main() == 0


def test_cli_validates_netcdf(tmp_path, monkeypatch):
    path = tmp_path / "scores.nc"
    make_metrics_dataset().to_netcdf(path)
    monkeypatch.setattr(
        "sys.argv", ["prog", "verification", "forecast_scoring_metrics", str(path)]
    )
    assert cli.main() == 0


def test_spec_text_renders_without_dataset():
    with skip_all_checks():
        report, spec_text = forecast_scoring_metrics.validate_dataset(None)
    assert report.results == []
    for heading in ["### 3.1", "### 3.3", "### 4.2", "### 4.4", "### 5.4"]:
        assert heading in spec_text


def test_reserved_dimension_fails():
    ds = make_metrics_dataset()
    ds = ds.expand_dims(variable=["tp"])
    assert "FAIL" in _statuses(_validate(ds), "3.1")


def test_missing_required_dimension_fails():
    ds = make_metrics_dataset().isel(init_time=0, drop=True)
    assert "FAIL" in _statuses(_validate(ds), "3.1")


def test_dimension_without_coordinate_fails():
    ds = make_metrics_dataset().expand_dims("threshold")
    ds = ds.drop_vars("threshold", errors="ignore")
    assert "FAIL" in _statuses(_validate(ds), "3.1")


def test_integer_lead_time_fails():
    ds = make_metrics_dataset()
    ds = ds.assign_coords(lead_time=np.arange(ds.lead_time.size) * 15)
    assert "FAIL" in _statuses(_validate(ds), "3.2")


def test_duplicate_init_time_fails():
    ds = make_metrics_dataset()
    init_time = ds.init_time.values.copy()
    init_time[1] = init_time[0]
    ds = ds.assign_coords(init_time=init_time)
    assert "FAIL" in _statuses(_validate(ds), "3.2")


def test_optional_dimension_types_are_checked():
    ds = make_metrics_dataset().expand_dims(spatial_region=[1, 2])
    assert "FAIL" in _statuses(_validate(ds), "3.2")


@pytest.mark.parametrize(
    "n_init_times, init_freq, status",
    [
        (40, "1D", "PASS"),
        (39, "1D", "WARNING"),
        (80, "12h", "PASS"),
        (60, "12h", "WARNING"),
    ],
)
def test_forecast_case_coverage(n_init_times, init_freq, status):
    ds = make_metrics_dataset(n_init_times=n_init_times, init_freq=init_freq)
    report = _validate(ds)
    coverage = [r for r in report.results if "coverage" in r.requirement]
    assert [r.status for r in coverage] == [status]


def test_unordered_init_time_warns():
    ds = make_metrics_dataset().isel(init_time=slice(None, None, -1))
    assert "WARNING" in _statuses(_validate(ds), "3.3")


def test_negative_lead_time_fails():
    ds = make_metrics_dataset()
    ds = ds.assign_coords(lead_time=ds.lead_time - pd.Timedelta("30min"))
    assert "FAIL" in _statuses(_validate(ds), "3.3")


@pytest.mark.parametrize("name", ["rmse", ".tp", "rmse."])
def test_bad_variable_name_fails(name):
    ds = make_metrics_dataset().rename({"rmse.tp": name})
    assert "FAIL" in _statuses(_validate(ds), "4.1")


def test_metric_with_dots_in_name_passes():
    ds = make_metrics_dataset().rename({"fss.tp": "fss.10km.tp"})
    report = _validate(ds)
    assert _statuses(report, "4.1") == {"PASS"}
    assert _statuses(report, "4.2") == {"PASS"}


def test_unknown_metric_without_polarity_fails():
    ds = make_metrics_dataset()
    del ds["fss.tp"].attrs["metric_polarity"]
    assert "FAIL" in _statuses(_validate(ds), "4.2")


def test_count_variable_fails():
    ds = make_metrics_dataset()
    ds["number_of_forecasts.tp"] = ds["rmse.tp"].notnull().astype(float)
    ds["number_of_forecasts.tp"].attrs = {"units": "1"}
    assert "FAIL" in _statuses(_validate(ds), "4.2")


@pytest.mark.parametrize("polarity", ["lower_is_better", "better"])
def test_bad_declared_polarity_fails(polarity):
    ds = make_metrics_dataset()
    ds["csi.tp"].attrs["metric_polarity"] = polarity
    assert "FAIL" in _statuses(_validate(ds), "4.2")


def test_integer_dtype_fails():
    ds = make_metrics_dataset()
    ds["rmse.tp"] = ds["rmse.tp"].astype(int)
    assert "FAIL" in _statuses(_validate(ds), "4.3")


def test_variable_missing_required_dim_fails():
    ds = make_metrics_dataset()
    ds["rmse.tp"] = ds["rmse.tp"].isel(lead_time=0, drop=True)
    assert "FAIL" in _statuses(_validate(ds), "4.3")


def test_forecast_source_without_finite_values_warns():
    ds = make_metrics_dataset()
    ds["rmse.tp"].loc[dict(forecast_source="model")] = np.nan
    assert "WARNING" in _statuses(_validate(ds), "4.3")


def test_missing_units_fails_and_missing_long_name_warns():
    ds = make_metrics_dataset()
    del ds["rmse.tp"].attrs["units"]
    del ds["csi.tp"].attrs["long_name"]
    statuses = _statuses(_validate(ds), "4.4")
    assert {"FAIL", "WARNING"} <= statuses


def test_missing_provenance_warns():
    ds = make_metrics_dataset()
    del ds.attrs["truth"]
    assert _statuses(_validate(ds), "5.1") == {"PASS", "WARNING"}


@pytest.mark.parametrize("allow_netcdf, status", [(True, "PASS"), (False, "FAIL")])
def test_netcdf_storage_only_when_allowed(tmp_path, allow_netcdf, status):
    path = tmp_path / "scores.nc"
    make_metrics_dataset().to_netcdf(path)
    report = check_zarr_format(
        xr.open_dataset(path),
        allowed_versions=[2, 3],
        require_consolidated_if_v2=True,
        allow_netcdf=allow_netcdf,
    )
    assert [r.status for r in report.results] == [status]


def test_in_section_relabels_results():
    report = ValidationReport()
    report.add("4.4", "requirement", "PASS")
    relabelled = report.in_section("1.2")
    assert [r.section for r in relabelled.results] == ["1.2"]
    assert [r.section for r in report.results] == ["4.4"]


def test_dataset_identifier_not_required():
    report = _validate(make_metrics_dataset())
    assert not [r for r in report.results if "identifier" in r.requirement]


def test_creator_provenance_not_required():
    ds = make_metrics_dataset()
    assert "mlcast_created_by" not in ds.attrs
    assert "mlcast_created_with" not in ds.attrs
    assert _statuses(_validate(ds), "5.4") == {"PASS"}


def test_created_with_any_github_repo_passes():
    ds = make_metrics_dataset()
    ds.attrs[
        "mlcast_created_with"
    ] = "https://github.com/leifdenby/mlcast-verification-tools@v0.1.0"
    assert _statuses(_validate(ds), "5.4") == {"PASS"}


@pytest.mark.parametrize(
    "attr, value",
    [
        ("mlcast_created_by", "Jane Doe"),
        (
            "mlcast_created_with",
            "https://github.com/leifdenby/mlcast-verification-tools",
        ),
    ],
)
def test_invalid_creator_provenance_fails_when_present(attr, value):
    ds = make_metrics_dataset()
    ds.attrs[attr] = value
    assert "FAIL" in _statuses(_validate(ds), "5.4")


def test_usable_by_mlwp_scorecard():
    scorecard = pytest.importorskip("mlwp_scorecard")
    scorecard.ScoreCard(
        make_metrics_dataset(),
        baseline="baseline",
        metric_polarity={"fss": "higher_is_better"},
    )
