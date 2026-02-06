from typing import Sequence

import fsspec
import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.3"


def has_consolidated_metadata(ds, storage_options=None):
    """
    Check whether a Zarr dataset opened via xarray has consolidated metadata.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset opened with `xr.open_zarr()`.
    storage_options : dict, optional
        The same storage_options that were used when opening the dataset.
        Required for remote stores (e.g. S3, GCS).

    Returns
    -------
    bool or None
        True if `.zmetadata` exists,
        False if not found,
        None if source path cannot be determined.
    """
    # Try to infer the original store path (xarray stores this in encoding)
    store_path = ds.encoding.get("source")
    if store_path is None:
        return None  # no source info (e.g. dataset from memory)

    # Create filesystem using same storage options as xarray
    fs, _, paths = fsspec.get_fs_token_paths(
        store_path, storage_options=storage_options
    )
    store_root = paths[0].rstrip("/")
    return fs.exists(f"{store_root}/.zmetadata")


@log_function_call
def check_zarr_format(
    ds: xr.Dataset,
    *,
    storage_options: dict = None,
    allowed_versions: Sequence[int],
    require_consolidated_if_v2: bool,
) -> ValidationReport:
    """Check Zarr format requirements."""
    report = ValidationReport()

    zarr_format = getattr(ds, "zarr_format", 2)  # Default to Zarr v2
    if zarr_format in allowed_versions:
        report.add(
            SECTION_ID,
            "Zarr version compatibility",
            "PASS",
            f"Using supported Zarr v{zarr_format} format",
        )
    else:
        report.add(
            SECTION_ID,
            "Zarr version compatibility",
            "FAIL",
            f"Unsupported Zarr version: v{zarr_format}",
        )

    if zarr_format == 2 and require_consolidated_if_v2:
        if has_consolidated_metadata(ds, storage_options=storage_options):
            report.add(
                SECTION_ID,
                "Consolidated metadata presence",
                "PASS",
                "Zarr v2 dataset has consolidated metadata",
            )
        else:
            report.add(
                SECTION_ID,
                "Consolidated metadata presence",
                "FAIL",
                "Zarr v2 dataset is missing consolidated metadata",
            )

    return report
