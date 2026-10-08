import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from ..data_vars_filter import iter_data_vars
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.1"


@log_function_call
def check_chunking_strategy(
    ds: xr.Dataset,
    time_chunksize: int,
    require_full_spatial_chunks: bool = False,
) -> ValidationReport:
    """
    Validate the chunking strategy of the dataset.

    Parameters:
        ds (xr.Dataset): The dataset to validate.
        time_chunksize (int): Required chunk size for the time dimension.
        require_full_spatial_chunks (bool): Require a single chunk along every
            dimension after time, so that each chunk covers the whole domain.

    Returns:
        ValidationReport: A report containing the results of the chunking strategy validation checks.
    """
    report = ValidationReport()

    for data_var, data_array in iter_data_vars(ds):
        if hasattr(data_array.data, "chunks"):
            chunks = data_array.data.chunks
            if len(chunks) >= 1 and all(c == time_chunksize for c in chunks[0]):
                report.add(
                    SECTION_ID,
                    f"Chunking strategy for {data_var}",
                    "PASS",
                    f"Correct chunking: {time_chunksize} chunk(s) per timestep",
                )
            else:
                report.add(
                    SECTION_ID,
                    f"Chunking strategy for {data_var}",
                    "FAIL",
                    f"Time dimension must be chunked as {time_chunksize} per timestep. Found: {chunks[0][:5]}...",
                )
            if require_full_spatial_chunks and data_array.ndim > 1:
                dims = data_array.dims[1:]
                domain = " × ".join(
                    f"{dim}={size}" for dim, size in zip(dims, data_array.shape[1:])
                )
                split = [
                    f"{dim} in {len(dim_chunks)} chunks"
                    for dim, dim_chunks in zip(dims, chunks[1:])
                    if len(dim_chunks) > 1
                ]
                if split:
                    report.add(
                        SECTION_ID,
                        f"Spatial chunking for {data_var}",
                        "FAIL",
                        f"Each chunk must cover the full domain ({domain}), found {', '.join(split)}",
                    )
                else:
                    report.add(
                        SECTION_ID,
                        f"Spatial chunking for {data_var}",
                        "PASS",
                        f"Each chunk covers the full domain ({domain})",
                    )
        else:
            report.add(
                SECTION_ID,
                f"Chunking strategy for {data_var}",
                "WARNING",
                "Data not chunked (not a dask array)",
            )

    return report
