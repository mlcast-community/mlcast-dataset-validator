import numpy as np
import xarray as xr

from mlcast_dataset_validator.checks.data_vars.chunking import check_chunking_strategy


def _dataset(chunks):
    data = np.zeros((4, 6, 8), dtype="float32")
    return xr.Dataset({"rr": (("time", "y", "x"), data)}).chunk(chunks)


def _statuses(report, requirement):
    return [r.status for r in report.results if r.requirement == requirement]


def test_full_domain_chunks_pass():
    report = check_chunking_strategy(
        _dataset({"time": 1}), time_chunksize=1, require_full_spatial_chunks=True
    )
    assert _statuses(report, "Spatial chunking for rr") == ["PASS"]
    assert not report.has_fails()


def test_spatially_tiled_chunks_fail():
    report = check_chunking_strategy(
        _dataset({"time": 1, "y": 3}),
        time_chunksize=1,
        require_full_spatial_chunks=True,
    )
    # one timestep per chunk is fine, but y is split in two tiles
    assert _statuses(report, "Chunking strategy for rr") == ["PASS"]
    assert _statuses(report, "Spatial chunking for rr") == ["FAIL"]


def test_spatial_check_off_by_default():
    report = check_chunking_strategy(_dataset({"time": 1, "y": 3}), time_chunksize=1)
    assert _statuses(report, "Spatial chunking for rr") == []
    assert not report.has_fails()
