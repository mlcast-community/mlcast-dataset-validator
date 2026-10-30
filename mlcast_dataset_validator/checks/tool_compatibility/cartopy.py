"""
Cartopy compatibility checks derived from the standalone tooling script supplied
alongside the radar source validator project.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from ..coords.names import find_cf_coordinates
from ..data_vars.georeferencing import NON_CF_CRS_ATTRS
from ..data_vars_filter import select_data_var
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.2"

try:  # pragma: no cover - optional dependency handling
    import cartopy.crs as ccrs  # type: ignore

    CARTOPY_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback
    CARTOPY_AVAILABLE = False
    ccrs = None  # type: ignore


def _select_data_variable(
    ds: xr.Dataset, preferred: Optional[str], role: Optional[str] = None
) -> Optional[str]:
    """Choose a data variable to use for sampling."""
    return select_data_var(ds, preferred, require_grid_mapping=True, role=role)


def build_cartopy_projection_from_cf(cf_attrs: dict) -> "ccrs.Projection":
    """
    Build a cartopy projection from CF grid mapping attributes.

    Cartopy cannot build a plottable projection from the WKT of some
    projections (e.g. geostationary, whose footprint is not a lat/lon box), so
    for those the dedicated cartopy class is constructed from the CF
    attributes instead.

    Parameters
    ----------
    cf_attrs : dict
        CF grid mapping attributes (``grid_mapping_name`` and the projection
        parameters, CF conventions appendix F).

    Returns
    -------
    ccrs.Projection
        The cartopy projection.

    Raises
    ------
    ValueError
        If ``grid_mapping_name`` is missing or not supported.
    """
    grid_mapping_name = cf_attrs.get("grid_mapping_name")
    if grid_mapping_name != "geostationary":
        raise ValueError(
            f"building a cartopy projection from CF attributes is not supported for "
            f"grid_mapping_name={grid_mapping_name!r} (supported: 'geostationary')"
        )
    globe = ccrs.Globe(
        semimajor_axis=cf_attrs.get("semi_major_axis"),
        semiminor_axis=cf_attrs.get("semi_minor_axis"),
        inverse_flattening=(
            None
            if "semi_minor_axis" in cf_attrs
            else cf_attrs.get("inverse_flattening")
        ),
        ellipse=None if "semi_major_axis" in cf_attrs else "WGS84",
    )
    return ccrs.Geostationary(
        central_longitude=float(cf_attrs.get("longitude_of_projection_origin", 0.0)),
        satellite_height=float(cf_attrs["perspective_point_height"]),
        false_easting=float(cf_attrs.get("false_easting", 0.0)),
        false_northing=float(cf_attrs.get("false_northing", 0.0)),
        sweep_axis=str(cf_attrs.get("sweep_angle_axis", "y")),
        globe=globe,
    )


def _read_valid_latlon_mask(
    ds: xr.Dataset, x_idx: np.ndarray, y_idx: np.ndarray
) -> Optional[np.ndarray]:
    """
    Read which sampled grid points have finite stored latitude/longitude.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with projected x/y and 2-D latitude/longitude coordinates.
    x_idx, y_idx : np.ndarray
        Indices along x and y of the sampled points.

    Returns
    -------
    np.ndarray or None
        Boolean mask with shape ``(len(y_idx), len(x_idx))``, or None if the
        dataset has no 2-D latitude/longitude over (y, x).
    """
    found = find_cf_coordinates(ds)
    if not (found["lat"] and found["lon"]):
        return None
    da_lat, da_lon = ds[found["lat"][0]], ds[found["lon"][0]]
    if set(da_lat.dims) != {"y", "x"} or set(da_lon.dims) != {"y", "x"}:
        return None
    mask = np.ones((len(y_idx), len(x_idx)), dtype=bool)
    for da in (da_lat, da_lon):
        values = da.isel(y=y_idx, x=x_idx).transpose("y", "x").values
        mask &= np.isfinite(values.astype(float))
    return mask


@log_function_call
def check_cartopy_compatibility(
    ds: xr.Dataset,
    *,
    data_variable: Optional[str] = None,
    crs_source: Literal["wkt", "cf"] = "wkt",
    role: Optional[str] = None,
) -> ValidationReport:
    """
    Validate Cartopy compatibility for a dataset.

    The check ensures that:
      1. Cartopy can instantiate a CRS object from the stored WKT string, or
         (with ``crs_source="cf"``) a plottable projection from the CF grid
         mapping attributes.
      2. The CRS description includes a BBOX (recommended for plotting; only
         checked for ``crs_source="wkt"``).
      3. Coordinate bounds exist and can be transformed to PlateCarree. Grid
         points whose stored latitude/longitude is not finite (e.g. off the
         disc of a geostationary grid) are expected to fail the transform and
         are not counted.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset opened via xarray.
    data_variable : str, optional
        Specific data variable to inspect; defaults to the first data variable.
    crs_source : {"wkt", "cf"}, optional
        Where to build the cartopy CRS from: the ``crs_wkt`` attribute
        (default) or the CF grid mapping attributes of the crs variable.
    role : str, optional
        Only consider data variables with this role (see
        :func:`~mlcast_dataset_validator.checks.data_vars_filter.classify_data_vars`),
        e.g. ``"observation"``.

    Returns
    -------
    ValidationReport
        Detailed PASS/FAIL/WARNING entries describing Cartopy compatibility.
    """
    report = ValidationReport()

    if not CARTOPY_AVAILABLE:
        report.add(
            SECTION_ID,
            "Cartopy availability",
            "WARNING",
            "cartopy is not installed; skipping cartopy-specific checks.",
        )
        return report

    target_var = _select_data_variable(ds, data_variable, role)
    if target_var is None:
        report.add(
            SECTION_ID,
            "Data variable selection",
            "FAIL",
            "No data variable with a 'grid_mapping' attribute is available for cartopy checks.",
        )
        return report

    data_array = ds[target_var]
    grid_mapping_name = data_array.attrs.get("grid_mapping")
    if not grid_mapping_name or grid_mapping_name not in ds.variables:
        report.add(
            SECTION_ID,
            "CRS metadata",
            "FAIL",
            f"Data variable '{target_var}' lacks a valid 'grid_mapping' attribute.",
        )
        return report

    crs_var = ds[grid_mapping_name]
    if crs_source == "cf":
        cf_attrs = {k: v for k, v in crs_var.attrs.items() if k not in NON_CF_CRS_ATTRS}
        try:
            crs = build_cartopy_projection_from_cf(cf_attrs)
            report.add(
                SECTION_ID,
                "Cartopy CRS creation",
                "PASS",
                f"Successfully created cartopy projection ({type(crs).__name__}) "
                f"from the CF grid mapping attributes of '{grid_mapping_name}'.",
            )
        except Exception as exc:
            report.add(
                SECTION_ID,
                "Cartopy CRS creation",
                "FAIL",
                f"Cartopy failed to build a projection from the CF grid mapping "
                f"attributes of '{grid_mapping_name}': {exc}",
            )
            return report
    else:
        wkt_string = crs_var.attrs.get("crs_wkt")
        if not wkt_string:
            report.add(
                SECTION_ID,
                "Cartopy CRS creation",
                "FAIL",
                f"CRS variable '{grid_mapping_name}' is missing 'crs_wkt' metadata.",
            )
            return report

        try:
            crs = ccrs.CRS(wkt_string)
            report.add(
                SECTION_ID,
                "Cartopy CRS creation",
                "PASS",
                f"Successfully created cartopy CRS instance ({type(crs).__name__}).",
            )
        except Exception as exc:
            report.add(
                SECTION_ID,
                "Cartopy CRS creation",
                "FAIL",
                f"Cartopy failed to parse CRS WKT: {exc}",
            )
            return report

        lower_wkt = wkt_string.lower()
        if "bbox" in lower_wkt:
            report.add(
                SECTION_ID,
                "Cartopy BBOX check",
                "PASS",
                "WKT definition includes a BBOX, aiding cartopy plotting.",
            )
        else:
            report.add(
                SECTION_ID,
                "Cartopy BBOX check",
                "WARNING",
                "WKT lacks a BBOX definition; cartopy plots may require manual extents.",
            )

    # Attempt to transform a handful of coordinates into PlateCarree for sanity.
    if {"x", "y"}.issubset(ds.coords):
        try:
            x_vals = ds.x.values
            y_vals = ds.y.values
            if len(x_vals) == 0 or len(y_vals) == 0:
                raise ValueError("Empty coordinate arrays.")

            x_idx = np.arange(len(x_vals))[:: max(1, len(x_vals) // 5)][:5]
            y_idx = np.arange(len(y_vals))[:: max(1, len(y_vals) // 5)][:5]
            xx, yy = np.meshgrid(x_vals[x_idx], y_vals[y_idx])

            plate_carree = ccrs.PlateCarree()
            transformed = plate_carree.transform_points(crs, xx.flatten(), yy.flatten())

            valid_mask = _read_valid_latlon_mask(ds, x_idx, y_idx)
            if valid_mask is not None and not valid_mask.all():
                # points without stored lat/lon (e.g. off the disc) are
                # expected not to transform
                transformed = transformed[valid_mask.flatten()]
                if transformed.shape[0] == 0:
                    raise ValueError(
                        "No sampled grid point has finite latitude/longitude."
                    )

            if not np.isfinite(transformed[:, :2]).all():
                report.add(
                    SECTION_ID,
                    "Cartopy coordinate transform",
                    "WARNING",
                    "Coordinate transformation produced NaN values.",
                )
            else:
                report.add(
                    SECTION_ID,
                    "Cartopy coordinate transform",
                    "PASS",
                    f"Successfully transformed {transformed.shape[0]} coordinate pairs to PlateCarree.",
                )
        except Exception as exc:
            report.add(
                SECTION_ID,
                "Cartopy coordinate transform",
                "FAIL",
                f"Failed to transform coordinates for cartopy plotting: {exc}",
            )
    else:
        report.add(
            SECTION_ID,
            "Cartopy coordinate transform",
            "WARNING",
            "Dataset lacks 'x' and 'y' coordinates; skipping transform test.",
        )

    return report
