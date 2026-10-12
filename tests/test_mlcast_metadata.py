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


def test_identifier_not_required_when_disabled():
    def identifier_results(**kwargs):
        report = check_mlcast_metadata(xr.Dataset(), **kwargs)
        return [r for r in report.results if "identifier" in r.requirement]

    assert any(r.status == "FAIL" for r in identifier_results())
    assert identifier_results(require_identifier=False) == []


def test_development_build_matches_its_release(monkeypatch):
    # builds from main between releases have versions like 0.4.0.dev3+g1234abc,
    # which should match datasets that conform to the upcoming 0.4.0 release
    monkeypatch.setattr(mlcast_metadata, "__version__", "0.4.0.dev3+g1234abc")
    result = _validator_version_results({VALIDATOR_VERSION_ATTRIBUTE: "0.4.0"})
    assert result.status == "PASS"
