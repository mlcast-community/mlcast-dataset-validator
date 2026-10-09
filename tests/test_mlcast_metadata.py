import pytest
import xarray as xr

from mlcast_dataset_validator.checks.global_attributes import mlcast_metadata
from mlcast_dataset_validator.checks.global_attributes.mlcast_metadata import (
    VALIDATOR_VERSION_ATTRIBUTE,
    check_mlcast_metadata,
)


def _validator_version_results(attrs):
    report = check_mlcast_metadata(xr.Dataset(attrs=attrs))
    results = [
        r for r in report.results if f"'{VALIDATOR_VERSION_ATTRIBUTE}'" in r.requirement
    ]
    assert len(results) == 1
    return results[0]


@pytest.fixture(autouse=True)
def running_version(monkeypatch):
    monkeypatch.setattr(mlcast_metadata, "__version__", "0.4.0")


def test_missing_validator_version_fails():
    result = _validator_version_results({})
    assert result.status == "FAIL"


def test_invalid_validator_version_fails():
    result = _validator_version_results({VALIDATOR_VERSION_ATTRIBUTE: "not a version"})
    assert result.status == "FAIL"


def test_matching_validator_version_passes():
    result = _validator_version_results({VALIDATOR_VERSION_ATTRIBUTE: "0.4.0"})
    assert result.status == "PASS"


def test_different_validator_version_warns():
    result = _validator_version_results({VALIDATOR_VERSION_ATTRIBUTE: "0.3.0"})
    assert result.status == "WARNING"
    assert "0.3.0" in result.detail and "0.4.0" in result.detail


def test_development_build_matches_its_release(monkeypatch):
    # builds from main between releases have versions like 0.4.0.dev3+g1234abc,
    # which should match datasets that conform to the upcoming 0.4.0 release
    monkeypatch.setattr(mlcast_metadata, "__version__", "0.4.0.dev3+g1234abc")
    result = _validator_version_results({VALIDATOR_VERSION_ATTRIBUTE: "0.4.0"})
    assert result.status == "PASS"


def _created_by_result(attrs):
    report = check_mlcast_metadata(xr.Dataset(attrs=attrs))
    results = [r for r in report.results if "'mlcast_created_by'" in r.requirement]
    assert len(results) == 1
    return results[0]


@pytest.mark.parametrize(
    "created_by",
    [
        "Ada Lovelace <ada@example.org>",
        "Ada Lovelace <ada@example.org>, Alan Turing <alan@example.org>",
        "  Ada Lovelace <ada@example.org> ,Alan Turing <alan@example.org>  ",
    ],
)
def test_valid_created_by_passes(created_by):
    result = _created_by_result({"mlcast_created_by": created_by})
    assert result.status == "PASS"


def test_multiple_creators_listed_in_detail():
    result = _created_by_result(
        {
            "mlcast_created_by": "Ada Lovelace <ada@example.org>, Alan Turing <alan@example.org>"
        }
    )
    assert "ada@example.org" in result.detail and "alan@example.org" in result.detail


@pytest.mark.parametrize(
    "created_by",
    [
        "Ada Lovelace <ada@example.org>, Alan Turing",
        "Ada Lovelace <ada@example.org>,",
        "Ada Lovelace <ada@example.org>,, Alan Turing <alan@example.org>",
        "ada@example.org",
    ],
)
def test_invalid_created_by_fails(created_by):
    result = _created_by_result({"mlcast_created_by": created_by})
    assert result.status == "FAIL"


def test_missing_created_by_fails():
    result = _created_by_result({})
    assert result.status == "FAIL"
