import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from ..data_vars_filter import iter_data_vars
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.1"


@log_function_call
def check_chunking_strategy(
    ds: xr.Dataset,
    time_chunksize: int,
) -> ValidationReport:
    """
    Validate the chunking strategy of the dataset.

    Parameters:
        ds (xr.Dataset): The dataset to validate.
        time_chunksize (int): Required chunk size for the time dimension.

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
        else:
            report.add(
                SECTION_ID,
                f"Chunking strategy for {data_var}",
                "WARNING",
                "Data not chunked (not a dask array)",
            )

    return report
