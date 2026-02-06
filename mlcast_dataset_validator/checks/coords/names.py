from typing import Dict, List, Sequence

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.1"

# Normalized unit strings accepted for latitude coordinates
_LAT_UNITS = {
    "degrees_north",
    "degree_north",
    "degrees_n",
    "degree_n",
    "deg_n",
}
# Normalized unit strings accepted for longitude coordinates
_LON_UNITS = {
    "degrees_east",
    "degree_east",
    "degrees_e",
    "degree_e",
    "deg_e",
}
# Linear distance units accepted for projected coordinates
_LINEAR_DISTANCE_UNITS = {
    "m",
    "meter",
    "meters",
    "metre",
    "metres",
    "km",
    "kilometer",
    "kilometers",
    "kilometre",
    "kilometres",
}

RuleSet = List[Dict[str, Sequence[str]]]

# Metadata-driven rules for identifying CF coordinate requirements.
# Each top-level key corresponds to a coordinate category (lat/lon/x/y/time).
# The value is a list of rule dictionaries; a coordinate matches the category
# if it satisfies *any* rule in the list (logical OR). Within each rule the
# coordinate must satisfy *all* attribute/value pairs (logical AND). Some
# rules check CF metadata (standard_name/units/axis) while others fall back
# to name-based heuristics when metadata is absent.
_COORD_RULES: Dict[str, RuleSet] = {
    "lat": [
        {"standard_name": ("latitude",)},
        {"axis": ("Y",), "units": tuple(_LAT_UNITS)},
        {"name": ("lat", "latitude")},
    ],
    "lon": [
        {"standard_name": ("longitude",)},
        {"axis": ("X",), "units": tuple(_LON_UNITS)},
        {"name": ("lon", "longitude")},
    ],
    "x": [
        {"standard_name": ("projection_x_coordinate",)},
        {"axis": ("X",), "units": tuple(_LINEAR_DISTANCE_UNITS)},
        {"name": ("x", "easting")},
    ],
    "y": [
        {"standard_name": ("projection_y_coordinate",)},
        {"axis": ("Y",), "units": tuple(_LINEAR_DISTANCE_UNITS)},
        {"name": ("y", "northing")},
    ],
    "time": [
        {"standard_name": ("time",)},
        {"axis": ("T",)},
        {"name": ("time",)},
    ],
}


def _normalize(value: str) -> str:
    """Lowercase-and-trim helper to normalize metadata values for comparison."""
    return value.strip().lower()


def _matches_rule(
    coord_name: str, coord_var: xr.DataArray, rule: Dict[str, Sequence[str]]
) -> bool:
    """
    Determine whether a coordinate satisfies a specific CF metadata rule.

    Parameters
    ----------
    coord_name : str
        Name of the coordinate variable.
    coord_var : xr.DataArray
        Coordinate data array with attached attributes.
    rule : dict
        Mapping of attribute names to allowed values (e.g., `standard_name`,
        `units`, `axis`, or synthetic `name`).

    Returns
    -------
    bool
        True if all required attributes match one of the allowed values.
    """
    for attr, allowed in rule.items():
        if attr == "name":
            value = coord_name
        else:
            value = coord_var.attrs.get(attr)

        if value is None:
            return False

        if attr == "axis":
            normalized_value = value.strip().upper()
            allowed_values = {str(option).strip().upper() for option in allowed}
        else:
            normalized_value = _normalize(str(value))
            allowed_values = {_normalize(str(option)) for option in allowed}

        if normalized_value not in allowed_values:
            return False

    return True


def _find_coordinates(ds: xr.Dataset, rules: RuleSet) -> List[str]:
    """
    Find coordinate names that satisfy at least one rule in the given set.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose coordinates should be inspected.
    rules : list[dict]
        Sequence of rule dictionaries describing acceptable metadata combos.

    Returns
    -------
    list[str]
        Names of coordinates that matched any rule.
    """
    matches: List[str] = []
    for coord_name in ds.coords:
        coord_var = ds.coords[coord_name]
        if any(_matches_rule(coord_name, coord_var, rule) for rule in rules):
            matches.append(coord_name)
    return matches


def _format_coord_list(names: Sequence[str]) -> str:
    """
    Render a list of coordinate names for human-readable reporting.

    Parameters
    ----------
    names : Sequence[str]
        Coordinate names to concatenate.

    Returns
    -------
    str
        Comma-separated list (or 'none' if empty).
    """
    if not names:
        return "none"
    return ", ".join(f"'{name}'" for name in names)


@log_function_call
def check_coordinate_names(
    ds: xr.Dataset,
    *,
    require_time_coord: bool = True,
    require_projected_coords: bool = False,
    require_latlon_coords: bool = False,
) -> ValidationReport:
    """
    Validate that the dataset exposes CF-compliant coordinate variables.

    The dataset must provide either:
      - Latitude/longitude coordinates with CF-compliant metadata
      - Projected x/y coordinates with CF-compliant metadata

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to evaluate.
    require_time_coord : bool, optional
        Require at least one CF-compliant time coordinate. Defaults to True.
    require_projected_coords : bool, optional
        Require projected x/y coordinates. Defaults to False.
    require_latlon_coords : bool, optional
        Require latitude/longitude coordinates. Defaults to False.
    """

    report = ValidationReport()

    time_coords = _find_coordinates(ds, _COORD_RULES["time"])
    if time_coords:
        report.add(
            SECTION_ID,
            "Time coordinate presence",
            "PASS",
            f"CF-compliant time coordinate(s) found: {_format_coord_list(time_coords)}",
        )
    elif require_time_coord:
        report.add(
            SECTION_ID,
            "Time coordinate presence",
            "FAIL",
            "Dataset is missing a CF-compliant time coordinate (requires `standard_name=time`, `axis=T`, or a 'time' coordinate).",
        )

    lat_coords = _find_coordinates(ds, _COORD_RULES["lat"])
    lon_coords = _find_coordinates(ds, _COORD_RULES["lon"])
    x_coords = _find_coordinates(ds, _COORD_RULES["x"])
    y_coords = _find_coordinates(ds, _COORD_RULES["y"])

    geographic_ok = bool(lat_coords and lon_coords)
    projected_ok = bool(x_coords and y_coords)

    if geographic_ok:
        report.add(
            SECTION_ID,
            "Latitude/longitude coordinates",
            "PASS",
            f"CF-compliant latitude ({_format_coord_list(lat_coords)}) and longitude ({_format_coord_list(lon_coords)}) coordinates detected.",
        )
    if projected_ok:
        report.add(
            SECTION_ID,
            "Projected coordinates",
            "PASS",
            f"CF-compliant projected x ({_format_coord_list(x_coords)}) and y ({_format_coord_list(y_coords)}) coordinates detected.",
        )

    failures: List[str] = []
    if require_latlon_coords and not geographic_ok:
        failures.append(
            "Latitude/longitude coordinates are required but no CF-compliant pair was found."
        )
    if require_projected_coords and not projected_ok:
        failures.append(
            "Projected x/y coordinates are required but no CF-compliant pair was found."
        )

    if not (geographic_ok or projected_ok):
        missing_geo = []
        if not lat_coords:
            missing_geo.append("latitude")
        if not lon_coords:
            missing_geo.append("longitude")
        missing_proj = []
        if not x_coords:
            missing_proj.append("projection_x_coordinate")
        if not y_coords:
            missing_proj.append("projection_y_coordinate")
        detail = (
            f"Latitude/longitude pair incomplete (missing CF-compliant {', '.join(missing_geo)} coordinate). "
            f"Projected x/y pair incomplete (missing CF-compliant {', '.join(missing_proj)} coordinate)."
        )
        failures.append(
            f"Dataset must include CF-compliant latitude/longitude or projected coordinates. {detail}"
        )

    if failures:
        report.add(
            SECTION_ID,
            "Coordinate reference compliance",
            "FAIL",
            " ".join(failures),
        )

    return report
