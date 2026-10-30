import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS, Transformer

# A small projected grid (UTM zone 32N) with CF-complete metadata, used by the
# coordinate attribute and georeferencing tests.
TEST_CRS = CRS.from_epsg(32632)


def make_projected_dataset(
    x_units: str = "m",
    n_x: int = 5,
    n_y: int = 4,
    crs_attr_overrides: dict | None = None,
    drop_crs_attrs: tuple = (),
    coord_attr_overrides: dict | None = None,
) -> xr.Dataset:
    """
    Build a dataset with projected x/y, 2-D lat/lon computed from TEST_CRS, a
    crs variable carrying WKT + CF grid mapping attributes and one data
    variable. ``x_units`` may be "m" or "km" (values are converted).
    """
    scale = 1000.0 if x_units == "km" else 1.0
    x_m = 500000.0 + np.arange(n_x) * 1000.0
    y_m = 5.0e6 - np.arange(n_y) * 1000.0
    xx, yy = np.meshgrid(x_m, y_m)
    lon, lat = Transformer.from_crs(TEST_CRS, "EPSG:4326", always_xy=True).transform(
        xx, yy
    )

    crs_attrs = TEST_CRS.to_cf()
    crs_attrs["spatial_ref"] = crs_attrs["crs_wkt"]
    crs_attrs.update(crs_attr_overrides or {})
    for key in drop_crs_attrs:
        crs_attrs.pop(key, None)

    coord_attrs = {
        "x": {
            "standard_name": "projection_x_coordinate",
            "units": x_units,
            "axis": "X",
        },
        "y": {
            "standard_name": "projection_y_coordinate",
            "units": x_units,
            "axis": "Y",
        },
        "lat": {"standard_name": "latitude", "units": "degrees_north"},
        "lon": {"standard_name": "longitude", "units": "degrees_east"},
    }
    for name, attrs in (coord_attr_overrides or {}).items():
        coord_attrs[name] = attrs

    time = pd.date_range("2020-01-01", periods=2, freq="h")
    ds = xr.Dataset(
        {
            "rr": (
                ("time", "y", "x"),
                np.zeros((2, n_y, n_x), dtype="float32"),
                {"grid_mapping": "crs"},
            ),
            "crs": ((), np.array(0, dtype="int32"), crs_attrs),
        },
        coords={
            "time": time,
            "x": ("x", x_m / scale, coord_attrs["x"]),
            "y": ("y", y_m / scale, coord_attrs["y"]),
            "lat": (("y", "x"), lat, coord_attrs["lat"]),
            "lon": (("y", "x"), lon, coord_attrs["lon"]),
        },
    )
    return ds


@pytest.fixture
def projected_dataset() -> xr.Dataset:
    return make_projected_dataset()


# The geostationary projection of MTG FCI, used by the geostationary satellite
# spec tests
GEOSTATIONARY_CF_ATTRS = {
    "grid_mapping_name": "geostationary",
    "latitude_of_projection_origin": 0.0,
    "longitude_of_projection_origin": 0.0,
    "perspective_point_height": 35786400.0,
    "sweep_angle_axis": "y",
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.31424518,
}
# Half-width (m) of the full-disc FCI grid in the geostationary projection
FULL_DISC_HALF_WIDTH_M = 5.5674e6


def make_geostationary_dataset(
    n: int = 300,
    step_m: float | None = 3000.0,
) -> xr.Dataset:
    """
    Build a small synthetic geostationary imager dataset of uint16 counts
    with slope/offset calibration variables.

    Parameters
    ----------
    n : int, optional
        Number of grid points along each of y and x. Defaults to 300.
    step_m : float or None, optional
        Grid spacing in metres, centred on the sub-satellite point. ``None``
        spreads the ``n`` points over the full disc, so that the corners are
        off the Earth and have NaN latitude/longitude. Defaults to 3000.

    Returns
    -------
    xr.Dataset
        Dataset meant to pass the `geostationary_satellite_counts` spec.
    """
    crs = CRS.from_cf(GEOSTATIONARY_CF_ATTRS)
    if step_m is None:
        x_m = np.linspace(-FULL_DISC_HALF_WIDTH_M, FULL_DISC_HALF_WIDTH_M, n)
    else:
        x_m = (np.arange(n) - (n - 1) / 2) * step_m
    y_m = x_m.copy()
    xx, yy = np.meshgrid(x_m, y_m)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xx, yy)
    lon = np.where(np.isfinite(lon) & np.isfinite(lat), lon, np.nan)
    lat = np.where(np.isfinite(lon), lat, np.nan)

    crs_attrs = dict(GEOSTATIONARY_CF_ATTRS)
    crs_attrs["crs_wkt"] = crs.to_wkt()
    crs_attrs["spatial_ref"] = crs_attrs["crs_wkt"]

    time = pd.to_datetime(["2025-01-01T00:00", "2026-01-02T00:00"])
    channels = ["vis_06", "ir_105"]
    shape = (len(time), n, n, len(channels))
    dims = ("time", "y", "x", "channel")

    data_vars = {
        "pixel_quality": (
            dims,
            np.zeros(shape, dtype="uint8"),
            {
                "long_name": "pixel quality flags",
                "units": "1",
                "grid_mapping": "spatial_ref",
                "flag_masks": [1, 2],
                "flag_meanings": "missing_warning saturation_warning",
            },
        ),
        "pixel_time": (
            dims,
            np.broadcast_to(time.values[:, None, None, None], shape).copy(),
            {
                "long_name": "per-pixel observation time",
                "standard_name": "time",
                "grid_mapping": "spatial_ref",
            },
        ),
        "spatial_ref": ((), np.array(0, dtype="int32"), crs_attrs),
        "counts": (
            dims,
            np.ones(shape, dtype="uint16"),
            {
                "long_name": "raw detector counts",
                "units": "1",
                "grid_mapping": "spatial_ref",
                "ancillary_variables": "pixel_quality pixel_time slope offset",
                "comment": "radiance = offset + slope * counts",
            },
        ),
    }
    for name in ("slope", "offset"):
        data_vars[name] = (
            ("time", "channel"),
            np.ones((len(time), len(channels)), dtype="float64"),
            {
                "long_name": f"calibration {name}",
                "units": "mW m-2 sr-1 (cm-1)-1",
            },
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": ("time", time, {"standard_name": "time", "axis": "T"}),
            "x": (
                "x",
                x_m,
                {
                    "standard_name": "projection_x_coordinate",
                    "units": "m",
                    "axis": "X",
                },
            ),
            "y": (
                "y",
                y_m,
                {
                    "standard_name": "projection_y_coordinate",
                    "units": "m",
                    "axis": "Y",
                },
            ),
            "latitude": (
                ("y", "x"),
                lat.astype("float32"),
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "longitude": (
                ("y", "x"),
                lon.astype("float32"),
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "channel": ("channel", channels, {"long_name": "channel name"}),
            "wavelength": (
                "channel",
                [0.64, 10.5],
                {
                    "standard_name": "radiation_wavelength",
                    "units": "um",
                    "long_name": "central wavelength",
                },
            ),
        },
        attrs={
            "license": "CC-BY-4.0",
            "platform": "MTG-I1",
            "instrument": "FCI",
            "mlcast_created_on": "2026-10-30T00:00:00",
            "mlcast_created_by": "Test Person <test@example.org>",
            "mlcast_created_with": "https://github.com/mlcast-community/mlcast-dataset-eumetsat-fci@v0.1.0",
            "mlcast_dataset_version": "0.1.0",
            "mlcast_dataset_validator_version": "0.4.0",
            "mlcast_dataset_identifier": "EU-EUMETSAT-counts",
        },
    )
    ds["pixel_time"].encoding.update(units="seconds since 2000-01-01", dtype="int64")
    return ds
