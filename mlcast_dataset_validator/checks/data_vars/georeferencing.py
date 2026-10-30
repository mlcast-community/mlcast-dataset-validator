from typing import Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from ...specs.reporting import ValidationReport, log_function_call
from ..coords.names import find_cf_coordinates
from ..data_vars_filter import select_data_vars_by_role
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.5"

# Attributes of a grid mapping variable that describe the CRS in another
# format. They are ignored when the CF attributes are validated on their own,
# because pyproj's ``CRS.from_cf`` silently prefers ``crs_wkt`` when present.
NON_CF_CRS_ATTRS = (
    "crs_wkt",
    "spatial_ref",
    "proj4",
    "projdef_original",
    "GeoTransform",
    "epsg_code",
)

_METRES_PER_UNIT = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "kilometre": 1000.0,
    "kilometres": 1000.0,
}


def cf_attrs_from_wkt(crs_wkt: str) -> dict:
    """CF grid mapping attributes equivalent to a WKT string (without ``crs_wkt``)."""
    attrs = CRS.from_wkt(crs_wkt).to_cf()
    attrs.pop("crs_wkt", None)
    return attrs


def _sample_indices(n: int) -> list:
    """Indices of the first, middle and last element (deduplicated, sorted)."""
    return sorted({0, n // 2, n - 1})


def _dense_sample_indices(n: int, count: int = 9) -> list:
    """Indices of ``count`` evenly spaced elements (deduplicated, sorted)."""
    return sorted({int(i) for i in np.linspace(0, n - 1, count).round()})


# Number of evenly spaced samples per axis used when some of the default
# samples have no valid latitude/longitude (e.g. off the disc of a
# geostationary full-disc grid)
_DENSE_SAMPLE_COUNT = 9


def _wrap_lon_diff(diff: np.ndarray) -> np.ndarray:
    """Wrap longitude differences into [-180, 180)."""
    return (diff + 180.0) % 360.0 - 180.0


def _max_deviation_from_latlon(
    ds: xr.Dataset,
    crs: CRS,
    coords: Tuple[str, str, str, str],
    x_scale: float,
    y_scale: float,
) -> float:
    """
    Max deviation (degrees) between lat/lon obtained by transforming sample
    x/y points through ``crs`` and the lat/lon stored in the dataset.

    ``x_scale``/``y_scale`` convert the coordinate values to the units of
    ``crs`` (e.g. 1000 for coordinates stored in km and a CRS in metres).

    Grid points whose stored latitude/longitude is not finite (e.g. off the
    disc of a geostationary full-disc grid) are left out of the comparison.
    If any of the default first/middle/last samples is such a point, a denser
    set of samples is used instead.
    """
    x_name, y_name, lat_name, lon_name = coords
    x_dim, y_dim = ds[x_name].dims[0], ds[y_name].dims[0]
    for name in (lat_name, lon_name):
        if set(ds[name].dims) != {x_dim, y_dim}:
            raise ValueError(
                f"'{name}' has dimensions {ds[name].dims}, expected 2-D over ({y_dim}, {x_dim})"
            )

    x = ds[x_name].values.astype(float) * x_scale
    y = ds[y_name].values.astype(float) * y_scale

    def _sample(ix: list, iy: list) -> Tuple[np.ndarray, ...]:
        px = np.array([x[i] for i in ix for j in iy])
        py = np.array([y[j] for i in ix for j in iy])
        lat_ref, lon_ref = (
            ds[name]
            .isel({y_dim: iy, x_dim: ix})
            .transpose(x_dim, y_dim)
            .values.ravel()
            .astype(float)
            for name in (lat_name, lon_name)
        )
        return px, py, lat_ref, lon_ref

    px, py, lat_ref, lon_ref = _sample(_sample_indices(len(x)), _sample_indices(len(y)))
    if not (np.isfinite(lat_ref).all() and np.isfinite(lon_ref).all()):
        px, py, lat_ref, lon_ref = _sample(
            _dense_sample_indices(len(x), _DENSE_SAMPLE_COUNT),
            _dense_sample_indices(len(y), _DENSE_SAMPLE_COUNT),
        )
        valid = np.isfinite(lat_ref) & np.isfinite(lon_ref)
        if not valid.any():
            raise ValueError(
                f"no sampled grid point has finite '{lat_name}'/'{lon_name}' values"
            )
        px, py, lat_ref, lon_ref = px[valid], py[valid], lat_ref[valid], lon_ref[valid]

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(px, py)
    deviations = np.concatenate(
        [
            np.abs(_wrap_lon_diff(np.asarray(lon) - lon_ref)),
            np.abs(np.asarray(lat) - lat_ref),
        ]
    )
    if np.isnan(deviations).any():
        raise ValueError("transformation produced NaN values")
    return float(deviations.max())


def _check_cf_grid_mapping(
    ds: xr.Dataset,
    data_var: str,
    grid_mapping: str,
    tolerance_deg: float,
) -> ValidationReport:
    """CF grid mapping attributes of ``grid_mapping`` and their consistency with lat/lon."""
    report = ValidationReport()
    crs_var = ds[grid_mapping]
    cf_attrs = {k: v for k, v in crs_var.attrs.items() if k not in NON_CF_CRS_ATTRS}
    wkt = crs_var.attrs.get("crs_wkt")

    hint = ""
    if wkt:
        try:
            hint = f" Attributes derived from crs_wkt: {cf_attrs_from_wkt(wkt)}"
        except Exception:  # unparsable WKT is reported by the tool compatibility checks
            hint = ""

    requirement = f"CF grid mapping for {data_var}"
    if "grid_mapping_name" not in cf_attrs:
        report.add(
            SECTION_ID,
            requirement,
            "FAIL",
            f"CRS variable '{grid_mapping}' has no 'grid_mapping_name' attribute (CF section 5.6)."
            + hint,
        )
        return report
    try:
        crs_cf = CRS.from_cf(cf_attrs)
    except Exception as exc:
        report.add(
            SECTION_ID,
            requirement,
            "FAIL",
            f"CF grid mapping attributes of '{grid_mapping}' cannot be interpreted: {exc}."
            + hint,
        )
        return report
    report.add(
        SECTION_ID,
        requirement,
        "PASS",
        f"'{grid_mapping}' defines grid_mapping_name='{cf_attrs['grid_mapping_name']}' "
        "with interpretable CF parameters",
    )

    # Consistency of the CF attributes (and of the WKT) with the stored lat/lon
    requirement = f"Georeferencing consistency for {data_var}"
    found = find_cf_coordinates(ds)
    if not all(found[c] for c in ("x", "y", "lat", "lon")):
        report.add(
            SECTION_ID,
            requirement,
            "WARNING",
            "Cannot compare with latitude/longitude: projected x/y or lat/lon coordinates not found",
        )
        return report
    coords = (found["x"][0], found["y"][0], found["lat"][0], found["lon"][0])
    scales = []
    for name in coords[:2]:
        units = str(ds[name].attrs.get("units", "m")).strip().lower()
        if units not in _METRES_PER_UNIT:
            report.add(
                SECTION_ID,
                requirement,
                "WARNING",
                f"Cannot compare with latitude/longitude: unrecognised units '{units}' on '{name}'",
            )
            return report
        scales.append(_METRES_PER_UNIT[units])

    descriptions = [("CF grid mapping attributes", crs_cf, scales[0], scales[1])]
    if wkt:
        try:
            # x/y are expressed in the units of the WKT's projected CRS, no scaling
            descriptions.append(("crs_wkt", CRS.from_wkt(wkt), 1.0, 1.0))
        except Exception:
            pass
    for label, crs, x_scale, y_scale in descriptions:
        try:
            deviation = _max_deviation_from_latlon(ds, crs, coords, x_scale, y_scale)
        except Exception as exc:
            report.add(
                SECTION_ID,
                requirement,
                "WARNING",
                f"Could not compare {label} with latitude/longitude: {exc}",
            )
            continue
        if deviation <= tolerance_deg:
            report.add(
                SECTION_ID,
                requirement,
                "PASS",
                f"{label} reproduce the stored latitude/longitude "
                f"(max deviation {deviation:.1e}° ≤ {tolerance_deg:g}°)",
            )
        else:
            report.add(
                SECTION_ID,
                requirement,
                "FAIL",
                f"{label} place the grid {deviation:.3g}° away from the stored latitude/longitude "
                f"(tolerance {tolerance_deg:g}°); check the projection parameters and the x/y units."
                + (hint if label != "crs_wkt" else ""),
            )
    return report


@log_function_call
def check_georeferencing(
    ds: xr.Dataset,
    *,
    require_geozarr: bool,
    require_grid_mapping: bool,
    crs_attrs: Sequence[str],
    require_bbox: bool,
    require_cf_grid_mapping: bool = False,
    latlon_tolerance_deg: float = 1e-4,
    roles: Optional[Sequence[str]] = None,
) -> ValidationReport:
    """
    Check georeferencing requirements.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to evaluate.
    require_geozarr, require_grid_mapping, require_bbox : bool
        Existing structural requirements (see the spec text).
    crs_attrs : Sequence[str]
        Attributes that must be present on the grid mapping variable.
    require_cf_grid_mapping : bool, optional
        Also require ``grid_mapping_name`` and CF projection parameters on the
        grid mapping variable (interpretable by pyproj without the WKT), and
        require both the CF attributes and ``crs_wkt`` to reproduce the stored
        latitude/longitude at sampled grid points. Defaults to False.
    latlon_tolerance_deg : float, optional
        Tolerance (degrees) for that comparison. Defaults to 1e-4 (about 10 m).
    roles : Sequence[str], optional
        Only check the data variables with these roles (see
        :func:`~mlcast_dataset_validator.checks.data_vars_filter.classify_data_vars`),
        e.g. to leave out calibration coefficients that have no spatial
        dimensions. By default all data variables are checked.
    """
    report = ValidationReport()
    if roles is not None:
        ds = select_data_vars_by_role(ds, roles)
    # Find all grid_mapping data variables since we don't want to check those
    grid_mapping_vars = set()
    for var in ds.data_vars:
        if "grid_mapping" in ds[var].attrs:
            grid_mapping_vars.add(ds[var].attrs["grid_mapping"])
    data_vars = set(ds.data_vars) - grid_mapping_vars

    for data_var in data_vars:
        data_array = ds[data_var]
        if require_grid_mapping and "grid_mapping" not in data_array.attrs:
            report.add(
                SECTION_ID,
                f"Grid mapping for {data_var}",
                "FAIL",
                f"Data variable '{data_var}' is missing 'grid_mapping' attribute",
            )
            continue

        grid_mapping = data_array.attrs.get("grid_mapping", None)
        if grid_mapping and grid_mapping in ds.variables:
            crs_var = ds[grid_mapping]
            missing_attrs = [attr for attr in crs_attrs if attr not in crs_var.attrs]
            if missing_attrs:
                report.add(
                    SECTION_ID,
                    f"CRS attributes for {data_var}",
                    "FAIL",
                    f"CRS variable '{grid_mapping}' is missing attributes: {missing_attrs}",
                )
            else:
                report.add(
                    SECTION_ID,
                    f"CRS attributes for {data_var}",
                    "PASS",
                    f"CRS variable '{grid_mapping}' has all required attributes",
                )
            if require_cf_grid_mapping:
                report += _check_cf_grid_mapping(
                    ds, data_var, grid_mapping, latlon_tolerance_deg
                )
        else:
            report.add(
                SECTION_ID,
                f"Grid mapping for {data_var}",
                "FAIL",
                f"Data variable '{data_var}' references a non-existent grid mapping variable",
            )

    return report
