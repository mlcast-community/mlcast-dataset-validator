import math
from typing import Sequence

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.2"


@log_function_call
def check_spatial_requirements(
    ds: xr.Dataset,
    max_resolution_km: float,
    rel_tol: float,
    min_crop_size: tuple[int, int],
    require_constant_domain: bool,
    spatial_dims: Sequence[str] | None = None,
) -> ValidationReport:
    """
    Validate spatial requirements for the dataset.

    Parameters:
        ds (xr.Dataset): The dataset to validate.
        max_resolution_km (float): Maximum allowed spatial resolution in kilometers.
        rel_tol (float): Maximum allowed relative tolerance for the spatial resolution.
        min_crop_size (tuple[int, int]): Minimum crop size as (height, width) in pixels.
        require_constant_domain (bool): Whether the spatial domain must remain constant across timesteps.
        spatial_dims (Sequence[str], optional): Names of the (height, width) spatial
            dimensions. When given, the crop size is measured over these dimensions
            only (other dimensions such as `channel` are ignored) and data variables
            that do not span them are skipped. When omitted, every dimension other
            than time is treated as spatial.

    Returns:
        ValidationReport: A report containing the results of the spatial validation checks.
    """
    report = ValidationReport()
    resolution_label = f"Spatial resolution ≤{max_resolution_km:g}km"
    crop_label = f"{min_crop_size[0]}×{min_crop_size[1]} pixel support"

    # Validate spatial resolution
    if "x" in ds.coords and "y" in ds.coords:
        try:
            x_vals = ds.x.values
            y_vals = ds.y.values
            if len(x_vals) > 1 and len(y_vals) > 1:
                x_res = abs(float(x_vals[1] - x_vals[0]))
                y_res = abs(float(y_vals[1] - y_vals[0]))
                max_resolution_m = max_resolution_km * 1000
                if (
                    x_res <= max_resolution_m
                    or math.isclose(x_res, max_resolution_m, rel_tol=rel_tol)
                ) and (
                    y_res <= max_resolution_m
                    or math.isclose(y_res, max_resolution_m, rel_tol=rel_tol)
                ):
                    report.add(
                        SECTION_ID,
                        resolution_label,
                        "PASS",
                        f"Resolution ({x_res:.1f}m × {y_res:.1f}m) ≤ {max_resolution_km}km",
                    )
                else:
                    report.add(
                        SECTION_ID,
                        resolution_label,
                        "FAIL",
                        f"Resolution ({x_res:.1f}m × {y_res:.1f}m) exceeds {max_resolution_km}km limit",
                    )
        except Exception as e:
            report.add(
                SECTION_ID,
                resolution_label,
                "WARNING",
                f"Could not verify spatial resolution: {e}",
            )

    # Find all grid_mapping data variables since we don't want to check those
    grid_mapping_vars = set()
    for var in ds.data_vars:
        if "grid_mapping" in ds[var].attrs:
            grid_mapping_vars.add(ds[var].attrs["grid_mapping"])
    data_vars = set(ds.data_vars) - grid_mapping_vars

    # Validate spatial coverage
    for data_var in sorted(data_vars):
        data_array = ds[data_var]
        dims = data_array.dims
        if spatial_dims is None:
            var_spatial_dims = [d for d in dims if d not in ["time", "t"]]
            if len(var_spatial_dims) < 2:
                report.add(
                    SECTION_ID,
                    "Spatial dimension check",
                    "FAIL",
                    f"Need at least 2 spatial dimensions for {data_var} ({dims})",
                )
                continue
            spatial_sizes = [data_array.sizes[d] for d in var_spatial_dims]
            supports_crop = all(s >= min_crop_size[0] for s in spatial_sizes)
        else:
            if not all(d in dims for d in spatial_dims):
                # e.g. per-channel calibration coefficients, not gridded data
                continue
            spatial_sizes = [data_array.sizes[d] for d in spatial_dims]
            supports_crop = all(
                s >= min_size for s, min_size in zip(spatial_sizes, min_crop_size)
            )
        if supports_crop:
            report.add(
                SECTION_ID,
                crop_label,
                "PASS",
                f"Spatial dimensions {spatial_sizes} support {min_crop_size[0]}×{min_crop_size[1]} crops",
            )
        else:
            report.add(
                SECTION_ID,
                crop_label,
                "FAIL",
                f"Spatial dimensions {spatial_sizes} too small for {min_crop_size[0]}×{min_crop_size[1]} crops",
            )

    return report
