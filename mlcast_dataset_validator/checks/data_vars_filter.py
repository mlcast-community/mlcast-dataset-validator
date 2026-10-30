from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

__all__ = [
    "DATA_VAR_ROLES",
    "classify_data_vars",
    "grid_mapping_definitions",
    "iter_data_vars",
    "select_data_var",
    "select_data_vars_by_role",
]

# Roles that `classify_data_vars` sorts data variables into
DATA_VAR_ROLES = (
    "observation",
    "quality_flag",
    "pixel_time",
    "calibration",
    "auxiliary",
)


def grid_mapping_definitions(ds: xr.Dataset) -> set[str]:
    """Return names of variables acting as CF grid_mapping definitions."""
    gm_vars: set[str] = set()
    for data_array in ds.data_vars.values():
        mapping_name = data_array.attrs.get("grid_mapping")
        if mapping_name and mapping_name in ds.variables:
            gm_vars.add(mapping_name)
    return gm_vars


def iter_data_vars(
    ds: xr.Dataset, *, require_grid_mapping: bool = False
) -> Iterator[Tuple[str, xr.DataArray]]:
    """Yield data variables, skipping grid_mapping definition variables."""
    gm_vars = grid_mapping_definitions(ds)
    for name, data_array in ds.data_vars.items():
        if name in gm_vars:
            continue
        if require_grid_mapping and "grid_mapping" not in data_array.attrs:
            continue
        yield name, data_array


def select_data_var(
    ds: xr.Dataset,
    preferred: Optional[str] = None,
    *,
    require_grid_mapping: bool = False,
    role: Optional[str] = None,
) -> Optional[str]:
    """
    Select a representative data variable respecting grid_mapping filters.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to select from.
    preferred : str, optional
        Name of the variable to use if it passes the filters.
    require_grid_mapping : bool, optional
        Only consider variables with a ``grid_mapping`` attribute.
    role : str, optional
        Only consider variables with this role (see :func:`classify_data_vars`).

    Returns
    -------
    str or None
        Name of the selected data variable, or None if none qualifies.
    """
    if role is not None:
        ds = select_data_vars_by_role(ds, [role])
    gm_vars = grid_mapping_definitions(ds)
    if preferred:
        if preferred in ds.data_vars and preferred not in gm_vars:
            if not require_grid_mapping or "grid_mapping" in ds[preferred].attrs:
                return preferred
        return None

    for name, _ in iter_data_vars(ds, require_grid_mapping=require_grid_mapping):
        return name
    return None


def classify_data_vars(
    ds: xr.Dataset, *, spatial_dims: Sequence[str] = ("y", "x")
) -> Dict[str, List[str]]:
    """
    Sort the data variables of a dataset into roles using their metadata.

    Grid mapping definition variables are left out. The roles are, in order of
    precedence:

    - ``quality_flag``: carries CF ``flag_masks`` or ``flag_values``
    - ``pixel_time``: has ``standard_name: time`` or a datetime dtype
    - ``observation``: spans all ``spatial_dims`` and is not referenced in the
      ``ancillary_variables`` attribute of another variable
    - ``calibration``: does not span the spatial dims and is referenced in the
      ``ancillary_variables`` attribute of another variable
    - ``auxiliary``: everything else

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose data variables are classified.
    spatial_dims : Sequence[str], optional
        Names of the spatial dimensions, by default ``("y", "x")``.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from each role in ``DATA_VAR_ROLES`` to the names of the data
        variables with that role (empty lists for roles without variables).
    """
    roles: Dict[str, List[str]] = {role: [] for role in DATA_VAR_ROLES}
    ancillary_refs = set()
    for da in ds.data_vars.values():
        ancillary_refs.update(str(da.attrs.get("ancillary_variables", "")).split())

    for name, da in iter_data_vars(ds):
        is_gridded = all(dim in da.dims for dim in spatial_dims)
        if "flag_masks" in da.attrs or "flag_values" in da.attrs:
            role = "quality_flag"
        elif da.attrs.get("standard_name") == "time" or np.issubdtype(
            da.dtype, np.datetime64
        ):
            role = "pixel_time"
        elif is_gridded and name not in ancillary_refs:
            role = "observation"
        elif not is_gridded and name in ancillary_refs:
            role = "calibration"
        else:
            role = "auxiliary"
        roles[role].append(name)
    return roles


def select_data_vars_by_role(
    ds: xr.Dataset,
    roles: Sequence[str],
    *,
    spatial_dims: Sequence[str] = ("y", "x"),
) -> xr.Dataset:
    """
    Select the data variables with the given roles.

    The grid mapping variables referenced by the selected variables are kept,
    as are the coordinates along their dimensions, so that checks of e.g.
    georeferencing still work on the selection.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to select from.
    roles : Sequence[str]
        Roles to select, from ``DATA_VAR_ROLES`` (see :func:`classify_data_vars`).
    spatial_dims : Sequence[str], optional
        Names of the spatial dimensions used for the classification, by
        default ``("y", "x")``.

    Returns
    -------
    xr.Dataset
        Dataset with only the selected data variables (and their grid mappings).

    Raises
    ------
    ValueError
        If an unknown role is given.
    """
    unknown = [role for role in roles if role not in DATA_VAR_ROLES]
    if unknown:
        raise ValueError(
            f"unknown data variable role(s) {unknown}, see {DATA_VAR_ROLES}"
        )
    classified = classify_data_vars(ds, spatial_dims=spatial_dims)
    selected = [name for role in roles for name in classified[role]]
    for name in list(selected):
        grid_mapping = ds[name].attrs.get("grid_mapping")
        if grid_mapping in ds.variables and grid_mapping not in selected:
            selected.append(grid_mapping)
    return ds[selected]
