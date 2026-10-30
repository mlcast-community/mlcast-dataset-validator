import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import make_geostationary_dataset

from mlcast_dataset_validator.checks.coords.channels import check_channel_coordinate
from mlcast_dataset_validator.checks.coords.spatial import check_spatial_requirements
from mlcast_dataset_validator.checks.coords.temporal import check_temporal_requirements
from mlcast_dataset_validator.checks.data_vars.calibration import (
    check_counts_calibration,
)
from mlcast_dataset_validator.checks.data_vars.data_structure import (
    check_data_structure,
)
from mlcast_dataset_validator.checks.data_vars.georeferencing import (
    check_georeferencing,
)
from mlcast_dataset_validator.checks.data_vars.naming import check_required_attrs
from mlcast_dataset_validator.checks.data_vars_filter import (
    classify_data_vars,
    select_data_var,
    select_data_vars_by_role,
)
from mlcast_dataset_validator.checks.global_attributes import mlcast_metadata
from mlcast_dataset_validator.checks.global_attributes.platform import (
    check_platform_attributes,
)
from mlcast_dataset_validator.checks.tool_compatibility import cartopy as cartopy_check
from mlcast_dataset_validator.checks.tool_compatibility.gdal import (
    _prepare_sample_slice,
)
from mlcast_dataset_validator.specs.source_data import geostationary_satellite_counts

SPATIAL_KWARGS = dict(
    max_resolution_km=3.0,
    rel_tol=1e-2,
    min_crop_size=(256, 256),
    require_constant_domain=True,
)


def _by_status(report, status):
    return [r for r in report.results if r.status == status]


@pytest.fixture
def ds_counts() -> xr.Dataset:
    return make_geostationary_dataset(n=8)


def test_classify_data_vars_roles(ds_counts):
    roles = classify_data_vars(ds_counts)
    assert roles["observation"] == ["counts"]
    assert roles["quality_flag"] == ["pixel_quality"]
    assert roles["pixel_time"] == ["pixel_time"]
    assert sorted(roles["calibration"]) == ["offset", "slope"]
    assert roles["auxiliary"] == []


def test_select_data_vars_by_role_keeps_grid_mapping(ds_counts):
    ds = select_data_vars_by_role(ds_counts, ["observation", "quality_flag"])
    assert sorted(ds.data_vars) == ["counts", "pixel_quality", "spatial_ref"]
    assert {"latitude", "longitude", "x", "y"} <= set(ds.coords)
    ds = select_data_vars_by_role(ds_counts, ["calibration"])
    assert sorted(ds.data_vars) == ["offset", "slope"]
    with pytest.raises(ValueError):
        select_data_vars_by_role(ds_counts, ["not_a_role"])


def test_select_data_var_by_role(ds_counts):
    # pixel_quality comes first in the dataset, but is not an observation
    assert list(ds_counts.data_vars)[0] == "pixel_quality"
    assert select_data_var(ds_counts, role="observation") == "counts"


def test_spatial_dims_ignores_channel_and_calibration_vars():
    ds = make_geostationary_dataset(n=300)
    report = check_spatial_requirements(ds, spatial_dims=("y", "x"), **SPATIAL_KWARGS)
    assert not report.has_fails()
    # without spatial_dims the channel dim is counted as spatial and the
    # (time, channel) calibration variables fail (as reported in
    # mlcast-datasets#43)
    report = check_spatial_requirements(ds, **SPATIAL_KWARGS)
    assert report.has_fails()


def test_crop_size_uses_spatial_dims_only():
    ds = make_geostationary_dataset(n=100)
    report = check_spatial_requirements(ds, spatial_dims=("y", "x"), **SPATIAL_KWARGS)
    fails = _by_status(report, "FAIL")
    assert fails and all("[100, 100]" in r.detail for r in fails)


def test_nat_in_time_coordinate_fails():
    times = pd.to_datetime(["2025-01-01", "NaT", "2025-01-03"])
    ds = xr.Dataset({"v": ("time", np.zeros(3))}, coords={"time": times})
    report = check_temporal_requirements(ds, min_years=0, allow_variable_timestep=True)
    fails = _by_status(report, "FAIL")
    assert len(fails) == 1
    assert "NaT" in fails[0].detail


def test_georeferencing_full_disc_skips_off_disc_points():
    ds = make_geostationary_dataset(n=21, step_m=None)
    assert np.isnan(ds.latitude.values[0, 0])  # corners are off the disc
    gridded = ["counts", "pixel_quality", "pixel_time", "spatial_ref"]
    report = check_georeferencing(
        ds[gridded],
        require_geozarr=True,
        require_grid_mapping=True,
        crs_attrs=["spatial_ref", "crs_wkt"],
        require_bbox=True,
        require_cf_grid_mapping=True,
    )
    assert not report.has_fails()
    assert not report.has_warnings()
    consistency = [
        r for r in _by_status(report, "PASS") if "consistency" in r.requirement
    ]
    # CF attributes and WKT compared for each of the three gridded variables
    assert len(consistency) == 6


@pytest.mark.skipif(not cartopy_check.CARTOPY_AVAILABLE, reason="needs cartopy")
def test_cartopy_projection_from_cf_attrs_on_full_disc():
    ds = make_geostationary_dataset(n=21, step_m=None)
    report = cartopy_check.check_cartopy_compatibility(
        ds, data_variable="counts", crs_source="cf"
    )
    assert not report.has_fails()
    statuses = {r.requirement: r.status for r in report.results}
    assert statuses["Cartopy CRS creation"] == "PASS"
    assert statuses["Cartopy coordinate transform"] == "PASS"


def test_channel_coordinate_passes(ds_counts):
    report = check_channel_coordinate(ds_counts)
    assert not report.has_fails()
    assert not report.has_warnings()


def test_channel_names_as_data_variable_fails_with_hint(ds_counts):
    ds = ds_counts.drop_vars(["channel", "wavelength"])
    ds["channel_name"] = ("channel", np.array([b"vis_06", b"ir_105"], dtype="S16"))
    report = check_channel_coordinate(ds)
    fails = _by_status(report, "FAIL")
    assert len(fails) == 1
    assert "set_coords('channel_name')" in fails[0].detail


def test_bytes_and_duplicate_channel_labels_fail(ds_counts):
    ds = ds_counts.assign_coords(channel=np.array([b"vis_06", b"ir_105"], dtype="S16"))
    assert "bytes" in _by_status(check_channel_coordinate(ds), "FAIL")[0].detail
    ds = ds_counts.assign_coords(channel=["ir_105", "ir_105"])
    assert "not unique" in _by_status(check_channel_coordinate(ds), "FAIL")[0].detail


def test_missing_wavelength_warns(ds_counts):
    report = check_channel_coordinate(ds_counts.drop_vars("wavelength"))
    assert not report.has_fails()
    assert report.has_warnings()


def test_counts_calibration_passes(ds_counts):
    report = check_counts_calibration(ds_counts)
    assert not report.has_fails()
    assert not report.has_warnings()


def test_counts_without_calibration_reference_fail(ds_counts):
    ds = ds_counts.copy()
    ds["counts"].attrs["ancillary_variables"] = "pixel_quality pixel_time"
    fails = _by_status(check_counts_calibration(ds), "FAIL")
    assert len(fails) == 1
    assert "calibration coefficient" in fails[0].detail


def test_float_counts_fail(ds_counts):
    ds = ds_counts.copy()
    ds["counts"] = ds["counts"].astype("float32")
    fails = _by_status(check_counts_calibration(ds), "FAIL")
    assert any("not as integers" in r.detail for r in fails)


def test_required_attrs_accepts_time_units_from_encoding(ds_counts):
    ds = ds_counts[["pixel_time"]]
    assert "units" not in ds.pixel_time.attrs
    assert not check_required_attrs(ds).has_fails()
    ds.pixel_time.encoding.pop("units")
    assert check_required_attrs(ds).has_fails()


def test_data_structure_can_check_encoded_dtype(ds_counts):
    ds = ds_counts[["counts"]].copy()
    ds["counts"] = ds["counts"].astype("float32")
    ds["counts"].encoding["dtype"] = np.dtype("uint16")
    kwargs = dict(dim_order=("time", "y", "x", "channel"), allowed_dtypes=["uint16"])
    assert check_data_structure(ds, **kwargs).has_fails()
    assert not check_data_structure(ds, use_encoded_dtype=True, **kwargs).has_fails()


def test_platform_attributes(ds_counts):
    assert not check_platform_attributes(ds_counts).has_fails()
    ds = ds_counts.copy()
    del ds.attrs["instrument"]
    assert len(_by_status(check_platform_attributes(ds), "FAIL")) == 1


def test_gdal_sample_slice_of_4d_variable(ds_counts):
    sample = _prepare_sample_slice(ds_counts["counts"])
    assert sample.dims == ("y", "x")
    assert sample.shape == (8, 8)


@pytest.fixture
def mock_github(monkeypatch):
    monkeypatch.setattr(mlcast_metadata, "__version__", "0.4.0")
    monkeypatch.setattr(mlcast_metadata, "github_repo_exists", lambda *a: True)
    monkeypatch.setattr(
        mlcast_metadata, "github_ref_exists", lambda *a: "/git/refs/tags/v0.1.0"
    )


def _open_as_zarr(ds: xr.Dataset, path) -> xr.Dataset:
    ds.chunk({"time": 1}).to_zarr(path, zarr_format=3, consolidated=False)
    return xr.open_zarr(path, consolidated=False)


def test_spec_passes_counts_dataset(tmp_path, mock_github):
    ds = _open_as_zarr(make_geostationary_dataset(), tmp_path / "ds.zarr")
    report, _ = geostationary_satellite_counts.validate_dataset(ds)
    assert _by_status(report, "FAIL") == []


def test_spec_fails_float_counts_without_calibration(tmp_path, mock_github):
    ds = make_geostationary_dataset()
    ds["counts"] = ds["counts"].astype("float32")
    ds["counts"].attrs["ancillary_variables"] = "pixel_quality pixel_time"
    ds = _open_as_zarr(ds, tmp_path / "ds.zarr")
    report, _ = geostationary_satellite_counts.validate_dataset(ds)
    failed_sections = {r.section for r in _by_status(report, "FAIL")}
    assert failed_sections == {"4.3", "4.6"}
