from typing import Sequence

import fsspec
import xarray as xr
from botocore.exceptions import NoCredentialsError

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.3"


def _has_consolidated_metadata(store_path: str, storage_options=None):
    """
    Check whether a Zarr dataset opened via xarray has consolidated metadata.

    Parameters
    ----------
    store_path : str
        Path/URL to the backing Zarr store.
    storage_options : dict, optional
        The same storage_options that were used when opening the dataset.
        Required for remote stores (e.g. S3, GCS).

    Returns
    -------
    bool
        ``True`` if `.zmetadata` exists at the store root, otherwise ``False``.

    """
    fs, _, paths = fsspec.get_fs_token_paths(
        store_path, storage_options=storage_options
    )
    store_root = paths[0].rstrip("/")
    return fs.exists(f"{store_root}/.zmetadata")


def _detect_zarr_format(store_path: str, storage_options=None):
    """
    Detect Zarr format version from the backing store.

    Parameters
    ----------
    store_path : str
        Path/URL to the backing Zarr store.
    storage_options : dict, optional
        Filesystem options used to access the store (for example S3/GCS
        credentials or endpoint settings). If omitted, fsspec defaults are used.

    Returns
    -------
    int
        Detected Zarr format version:
        - ``3`` when ``zarr.json`` exists at the store root.
        - ``2`` otherwise.

    """
    fs, _, paths = fsspec.get_fs_token_paths(
        store_path, storage_options=storage_options
    )
    store_root = paths[0].rstrip("/")

    if fs.exists(f"{store_root}/zarr.json"):
        return 3
    return 2


def _is_remote_store(store_path: str) -> bool:
    """
    Determine whether a store path refers to a remote filesystem.

    Parameters
    ----------
    store_path : str
        Path/URL to the backing Zarr store.

    Returns
    -------
    bool
        ``True`` when the inferred protocol is not local file-based, otherwise ``False``.
    """
    protocol = fsspec.utils.infer_storage_options(store_path).get("protocol")
    return protocol not in (None, "file", "local")


@log_function_call
def check_zarr_format(
    ds: xr.Dataset,
    *,
    allowed_versions: Sequence[int],
    require_consolidated_if_v2: bool,
) -> ValidationReport:
    """
    Validate Zarr format compatibility and consolidated metadata requirements.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset under validation.
    allowed_versions : Sequence[int]
        Zarr format versions accepted by the specification.
    require_consolidated_if_v2 : bool
        Whether Zarr v2 datasets must include consolidated metadata.

    Returns
    -------
    ValidationReport
        Report containing PASS/FAIL/WARNING entries for Zarr version checks and,
        when applicable, consolidated metadata validation.
    """
    report = ValidationReport()

    store_path = ds.encoding.get("source")
    if not store_path:
        report.add(
            SECTION_ID,
            "Zarr version compatibility",
            "FAIL",
            "Cannot determine Zarr version because dataset source path is missing.",
        )
        if require_consolidated_if_v2:
            report.add(
                SECTION_ID,
                "Consolidated metadata presence",
                "FAIL",
                "Cannot verify consolidated metadata because dataset source path is missing.",
            )
        return report

    storage_options = ds.encoding.get("storage_options")
    try:
        zarr_format = _detect_zarr_format(store_path, storage_options)
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
            consolidated = _has_consolidated_metadata(
                store_path, storage_options=storage_options
            )
            if consolidated is True:
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
    # not providing anon=True in storage_options for an S3 store without
    # credentials will raise a NoCredentialsError, while a local store that
    # doesn't exist will raise a FileNotFoundError
    except (FileNotFoundError, NoCredentialsError) as exc:
        if _is_remote_store(store_path):
            report.add(
                SECTION_ID,
                "Zarr store accessibility",
                "WARNING",
                (
                    "Could not access dataset store for Zarr checks. "
                    "Since this appears to be a remote store, your may need to "
                    "set any `storage_options` that you provided to xr.open_zarr(...) "
                    "as ds.encoding['storage_options'] so that validation checks can access the store. "
                    f"({exc})"
                ),
            )
        report.add(
            SECTION_ID,
            "Zarr version compatibility",
            "FAIL",
            "Cannot determine Zarr version because dataset store is not accessible.",
        )
        if require_consolidated_if_v2:
            report.add(
                SECTION_ID,
                "Consolidated metadata presence",
                "FAIL",
                "Cannot verify consolidated metadata because dataset store is not accessible.",
            )

    return report
