# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "xarray", "netcdf4", "zarr"]
# ///
"""
Write a small example dataset conforming to the MLCast
`verification/forecast_scoring_metrics` specification, e.g. to try out the validator or
as a template for writing your own verification scores:

    uv run tests/scripts/make_verification_scores_example.py scores.nc
    uvx --from mlcast-dataset-validator mlcast.validate_dataset verification forecast_scoring_metrics scores.nc

The scores are random numbers, only the layout and metadata are meaningful.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_VALIDATOR_VERSION = "0.4.0"


def _validator_version() -> str:
    """Release version of the installed mlcast-dataset-validator, if any."""
    try:
        from packaging.version import Version

        from mlcast_dataset_validator import __version__
    except ImportError:
        return DEFAULT_VALIDATOR_VERSION
    return Version(__version__).base_version


def make_verification_scores_example(
    n_init_times: int = 40,
    init_freq: str = "1D",
    validator_version: str | None = None,
) -> xr.Dataset:
    """
    Create an example forecast scoring metrics dataset.

    The dataset has scores for two forecast sources ("baseline" and "model")
    at 8 lead times (15 min to 2 h), for `n_init_times` forecast cases (§3),
    of three metrics of precipitation (`tp`, §4): `rmse.tp` and `csi.tp`,
    whose polarity is known, and `fss.tp`, which declares its polarity in the
    `metric_polarity` attribute. It also carries the provenance, license and
    MLCast metadata global attributes (§5).

    Parameters
    ----------
    n_init_times : int, optional
        Number of forecast cases (initialisation times). The default of 40
        daily cases is the fewest that give 4 blocks of 10 days (§3.3).
    init_freq : str, optional
        pandas frequency string of the initialisation times.
    validator_version : str, optional
        Value of the `mlcast_dataset_validator_version` attribute. Defaults to
        the release version of the installed mlcast-dataset-validator, or
        `DEFAULT_VALIDATOR_VERSION` if it isn't installed.

    Returns
    -------
    xr.Dataset
        The example dataset.
    """
    dims = ("forecast_source", "lead_time", "init_time")
    coords = dict(
        forecast_source=["baseline", "model"],
        lead_time=pd.timedelta_range("15min", periods=8, freq="15min"),
        init_time=pd.date_range("2024-01-01", periods=n_init_times, freq=init_freq),
    )
    rng = np.random.default_rng(0)
    shape = tuple(len(coords[dim]) for dim in dims)

    return xr.Dataset(
        {
            "rmse.tp": (dims, rng.random(shape), {"units": "mm", "long_name": "RMSE"}),
            "csi.tp": (dims, rng.random(shape), {"units": "1", "long_name": "CSI"}),
            "fss.tp": (
                dims,
                rng.random(shape),
                {
                    "units": "1",
                    "long_name": "FSS",
                    "metric_polarity": "higher_is_better",
                },
            ),
        },
        coords=coords,
        attrs={
            "title": "Example precipitation nowcast scores",
            "forecasts": "a baseline and a model",
            "truth": "radar composite",
            "license": "CC-BY-4.0",
            "mlcast_created_on": "2026-10-12T00:00:00",
            "mlcast_dataset_version": "0.1.0",
            "mlcast_dataset_validator_version": validator_version
            or _validator_version(),
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="scores.nc",
        help="Output path, written as Zarr if it ends in .zarr, else as netCDF",
    )
    parser.add_argument("--n-init-times", type=int, default=40)
    parser.add_argument("--init-freq", default="1D")
    args = parser.parse_args(argv)

    ds = make_verification_scores_example(
        n_init_times=args.n_init_times, init_freq=args.init_freq
    )
    if args.output.endswith(".zarr"):
        ds.to_zarr(args.output, mode="w")
    else:
        ds.to_netcdf(args.output)

    print(f"Wrote {args.output}, validate it with:")
    print(
        f"  mlcast.validate_dataset verification forecast_scoring_metrics {args.output}"
    )


if __name__ == "__main__":
    main()
