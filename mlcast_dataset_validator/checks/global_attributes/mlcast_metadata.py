"""Global attribute validators for MLCast metadata."""

import string
from typing import Dict

import isodate
import requests
import xarray as xr
from isodate import ISO8601Error, parse_datetime
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version
from parse import compile as parse_compile
from requests import RequestException

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.4"
CREATED_BY_FORMAT = "{name} <{email}>"
GITHUB_WITH_VERSION_FORMAT = "https://github.com/{org}/{repo}@{version}"
DEFAULT_DATASET_IDENTIFIER_FORMAT = "{country_code}-{entity}-{physical_variable}"
DATASET_IDENTIFIER_ATTRIBUTE = "mlcast_dataset_identifier"
DATASET_IDENTIFIER_FORMAT_ATTRIBUTE = "mlcast_dataset_identifier_format"
EXPECTED_GITHUB_ORG = "mlcast-community"
EXPECTED_REPO_PATTERN = "mlcast-dataset-{organisation_id}-{dataset_name}"


def is_alphanumeric(value: str) -> bool:
    """
    Validate a dataset identifier token string.

    Parameters
    ----------
    value : str
        Token value to validate; must be non-empty, contain no whitespace, use only
        alphanumerics plus '_' (hyphens are reserved as separators), and not start
        or end with '_'.

    Returns
    -------
    bool
        True if valid.

    Raises
    ------
    ValueError
        If the token is invalid.
    """
    if not value:
        raise ValueError("identifier token is empty")
    if any(ch.isspace() for ch in value):
        raise ValueError(f"identifier token contains whitespace: '{value}'")
    if value[0] == "_" or value[-1] == "_":
        raise ValueError(f"identifier token cannot start/end with '_': '{value}'")
    if not all(ch.isalnum() or ch == "_" for ch in value):
        raise ValueError(f"identifier token contains invalid characters: '{value}'")
    return True


def is_country_code(value: str) -> bool:
    """
    Validate an ISO 3166-1 alpha-2 style country code.

    Parameters
    ----------
    value : str
        Country code to validate.

    Returns
    -------
    bool
        True if valid.

    Raises
    ------
    ValueError
        If the country code is invalid.
    """
    if len(value) != 2:
        raise ValueError("country code must be 2 uppercase letters")
    if not value.isalpha() or not value.isupper():
        raise ValueError("country code must use uppercase alphabetic characters")
    return True


IDENTIFIER_PART_VALIDATORS = {
    "country_code": is_country_code,
    "entity": is_alphanumeric,
    "physical_variable": is_alphanumeric,
    "time_resolution": isodate.parse_duration,
    "common_name": is_alphanumeric,
}

_CREATED_BY_PARSER = parse_compile(CREATED_BY_FORMAT)
_GITHUB_WITH_VERSION_PARSER = parse_compile(GITHUB_WITH_VERSION_FORMAT)
_EXPECTED_REPO_PARSER = parse_compile(EXPECTED_REPO_PATTERN)
_GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "mlcast-dataset-validator",
}


def parse_created_by(value: str) -> Dict[str, str]:
    """
    Parse creator info in 'Name <email>' format.

    Parameters
    ----------
    value : str
        Creator string to parse.

    Returns
    -------
    dict
        Dictionary with keys 'name' and 'email'.

    Raises
    ------
    ValueError
        If the value does not match or has invalid components.
    """
    try:
        parsed = _CREATED_BY_PARSER.parse(value.strip())
    except AttributeError:
        raise ValueError("Value is not a string")
    if not parsed:
        raise ValueError("Does not match 'Name <email>'")
    name = parsed["name"].strip()
    email = parsed["email"].strip()
    if not name:
        raise ValueError("Missing name")
    if not email or "@" not in email:
        raise ValueError("Missing or invalid email")
    if any(ch.isspace() for ch in email):
        raise ValueError("Email contains whitespace")
    return {"name": name, "email": email}


def parse_github_url_with_version(value: str) -> Dict[str, str]:
    """
    Parse GitHub URL with version suffix and enforce mlcast conventions.

    Parameters
    ----------
    value : str
        URL string to parse.

    Returns
    -------
    dict
        Dictionary with keys 'org', 'repo', and 'version'.

    Raises
    ------
    ValueError
        If the value does not match or has invalid components.
    """
    try:
        parsed = _GITHUB_WITH_VERSION_PARSER.parse(value.strip())
    except AttributeError:
        raise ValueError("Value is not a string")
    if not parsed:
        raise ValueError("Does not match https://github.com/{org}/{repo}@{version}")
    org = parsed["org"].strip()
    repo = parsed["repo"].strip()
    version = parsed["version"].strip()
    if not (org and repo and version):
        raise ValueError("Missing org, repo, or version")
    if any(any(ch.isspace() for ch in part) for part in (org, repo, version)):
        raise ValueError("Contains whitespace in org/repo/version")
    if org != EXPECTED_GITHUB_ORG:
        raise ValueError(f"GitHub organisation must be '{EXPECTED_GITHUB_ORG}'")
    parsed_repo = _EXPECTED_REPO_PARSER.parse(repo)
    if not parsed_repo:
        raise ValueError(f"Repository must follow '{EXPECTED_REPO_PATTERN}'")
    org_id = parsed_repo["organisation_id"]
    dataset_name = parsed_repo["dataset_name"]
    if not org_id or not dataset_name:
        raise ValueError(
            "Repository must include non-empty organisation_id and dataset_name"
        )
    return {"org": org, "repo": repo, "version": version}


def github_repo_exists(org: str, repo: str) -> bool:
    """
    Return True if GitHub repository exists, False if 404, else raise on other issues.

    Parameters
    ----------
    org : str
        GitHub organisation name.
    repo : str
        GitHub repository name.

    Returns
    -------
    bool
        True if repository exists, False if not found.

    Raises
    ------
    RuntimeError
        If the GitHub API returns an unexpected response.
    """
    base = f"https://api.github.com/repos/{org}/{repo}"
    response = requests.get(base, headers=_GITHUB_HEADERS, timeout=5)
    if response.status_code == 404:
        return False
    if not response.ok:
        raise RuntimeError(
            f"GitHub repo check failed with status {response.status_code}"
        )
    return True


def github_ref_exists(org: str, repo: str, ref: str) -> str | None:
    """
    Check if a GitHub revision exists (tag, branch, or commit).

    Returns the matched API path if found, None if not found, otherwise raises.

    Parameters
    ----------
    org : str
        GitHub organisation name.
    repo : str
        GitHub repository name.
    ref : str
        Revision identifier (tag, branch, or commit hash).

    Returns
    -------
    str or None
        The matched API path if found, None if not found.

    Raises
    ------
    RuntimeError
        If the GitHub API returns an unexpected response.
    """
    base = f"https://api.github.com/repos/{org}/{repo}"
    ref_paths = [
        f"/git/refs/tags/{ref}",
        f"/git/refs/heads/{ref}",
        f"/commits/{ref}",
    ]
    for path in ref_paths:
        response = requests.get(base + path, headers=_GITHUB_HEADERS, timeout=5)
        if response.status_code == 404:
            continue
        if response.ok:
            return path
        raise RuntimeError(
            f"GitHub ref check failed with status {response.status_code} for {path}"
        )
    return None


def extract_identifier_format_fields(format_str: str) -> list[str]:
    """
    Extract ordered field names from a format string.

    Parameters
    ----------
    format_str : str
        Format string to parse.

    Returns
    -------
    list[str]
        Ordered field names extracted from the format string.

    Raises
    ------
    ValueError
        If the format string contains unsupported fields or specifiers.
    """
    formatter = string.Formatter()
    fields: list[str] = []
    for _, field_name, format_spec, conversion in formatter.parse(format_str):
        if field_name is None:
            continue
        if not field_name:
            raise ValueError("Empty field name in format string")
        if format_spec or conversion:
            raise ValueError("Format specifiers and conversions are not supported")
        fields.append(field_name)
    return fields


def validate_identifier_format(format_str: str) -> list[str]:
    """
    Validate identifier format and return ordered field names.

    Parameters
    ----------
    format_str : str
        Format string to validate.

    Returns
    -------
    list[str]
        Ordered field names extracted from the format string.

    Raises
    ------
    ValueError
        If the format string is invalid.
    """
    if not isinstance(format_str, str):
        raise ValueError("Value is not a string")
    trimmed = format_str.strip()
    if not trimmed:
        raise ValueError("Format string cannot be empty")
    if not trimmed.startswith(DEFAULT_DATASET_IDENTIFIER_FORMAT):
        raise ValueError(
            f"Format must start with '{DEFAULT_DATASET_IDENTIFIER_FORMAT}'"
        )
    fields = extract_identifier_format_fields(trimmed)
    required_parts = {"country_code", "entity", "physical_variable"}
    missing = required_parts.difference(fields)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Format missing required parts: {missing_list}")
    unknown = [field for field in fields if field not in IDENTIFIER_PART_VALIDATORS]
    if unknown:
        unknown_list = ", ".join(sorted(set(unknown)))
        raise ValueError(f"Format contains unsupported parts: {unknown_list}")
    duplicates = [field for field in set(fields) if fields.count(field) > 1]
    if duplicates:
        dup_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Format contains duplicate parts: {dup_list}")
    return fields


def parse_dataset_identifier(value: str, format_str: str) -> Dict[str, str]:
    """
    Parse dataset identifier using the provided format string.

    Parameters
    ----------
    value : str
        Dataset identifier to parse.
    format_str : str
        Format string to parse with.

    Returns
    -------
    dict
        Dictionary with parsed identifier parts.

    Raises
    ------
    ValueError
        If the value does not match or has invalid components.
    """
    try:
        parser = parse_compile(format_str)
    except Exception as exc:
        raise ValueError(f"Could not compile format: {exc}")
    try:
        parsed = parser.parse(value.strip())
    except AttributeError:
        raise ValueError("Value is not a string")
    if not parsed:
        raise ValueError("Does not match dataset identifier format")
    parts = {key: str(val).strip() for key, val in parsed.named.items()}
    for key, validator in IDENTIFIER_PART_VALIDATORS.items():
        if key in parts:
            try:
                validator(parts[key])
            except (ISO8601Error, ValueError, TypeError) as exc:
                raise ValueError(f"{key} {exc}") from exc
    return parts


@log_function_call
def check_mlcast_metadata(ds: xr.Dataset) -> ValidationReport:
    """
    Validate required MLCast-specific global attributes.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to validate.

    Returns
    -------
    ValidationReport
        Validation report with metadata checks.

    Required global attributes
    --------------------------
    mlcast_created_on
        ISO 8601 datetime string for dataset creation.
    mlcast_created_by
        Creator contact in 'Name <email>' format.
    mlcast_created_with
        GitHub URL of the creating software including version.
    mlcast_dataset_version
        Dataset specification version (semver or calver).
    mlcast_dataset_identifier
        Dataset identifier matching the dataset identifier format rules.

    Optional global attributes
    --------------------------
    mlcast_dataset_identifier_format
        Custom identifier format string. Must start with the default
        '{country_code}-{entity}-{physical_variable}' and only use approved parts
        listed in IDENTIFIER_PART_VALIDATORS.
    """
    report = ValidationReport()
    attrs = ds.attrs

    created_on = attrs.get("mlcast_created_on")
    if created_on is None:
        report.add(
            SECTION_ID,
            "Global attribute 'mlcast_created_on'",
            "FAIL",
            "Missing required creation timestamp in ISO 8601 format",
        )
    else:
        try:
            parsed_dt = parse_datetime(created_on.strip())
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_created_on'",
                "PASS",
                f"Creation timestamp parsed ({parsed_dt})",
            )
        except (ISO8601Error, ValueError, TypeError, AttributeError) as exc:
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_created_on'",
                "FAIL",
                f"Value '{created_on}' is not a valid ISO 8601 datetime string: {exc}",
            )

    created_by = attrs.get("mlcast_created_by")
    if created_by is None:
        report.add(
            SECTION_ID,
            "Global attribute 'mlcast_created_by'",
            "FAIL",
            "Missing required creator contact in 'Name <email>' format",
        )
    else:
        try:
            parsed_by = parse_created_by(created_by)
            detail = (
                f"Creator contact present ({parsed_by['name']} <{parsed_by['email']}>)"
            )
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_created_by'",
                "PASS",
                detail,
            )
        except ValueError as exc:
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_created_by'",
                "FAIL",
                f"Creator contact '{created_by}' is not in 'Name <email>' format: {exc}",
            )

    created_with = attrs.get("mlcast_created_with")
    if created_with is None:
        report.add(
            SECTION_ID,
            "Global attribute 'mlcast_created_with'",
            "FAIL",
            "Missing required creator software GitHub URL with version (e.g. https://github.com/org/repo@v0.1.0)",
        )
    else:
        try:
            parsed_created_with = parse_github_url_with_version(created_with)
            url_detail = (
                f"Creator software GitHub URL parsed "
                f"({parsed_created_with['org']}/{parsed_created_with['repo']}@{parsed_created_with['version']})"
            )
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_created_with'",
                "PASS",
                url_detail,
            )
            try:
                repo_exists = github_repo_exists(
                    parsed_created_with["org"], parsed_created_with["repo"]
                )
                if repo_exists:
                    report.add(
                        SECTION_ID,
                        "Creator software GitHub repository",
                        "PASS",
                        "Repository exists on GitHub",
                    )
                    try:
                        ref_path = github_ref_exists(
                            parsed_created_with["org"],
                            parsed_created_with["repo"],
                            parsed_created_with["version"],
                        )
                        if ref_path:
                            report.add(
                                SECTION_ID,
                                "Creator software GitHub revision",
                                "PASS",
                                f"Revision found at {ref_path}",
                            )
                        else:
                            report.add(
                                SECTION_ID,
                                "Creator software GitHub revision",
                                "FAIL",
                                "Revision not found as tag, branch, or commit",
                            )
                    except RequestException as exc:
                        report.add(
                            SECTION_ID,
                            "Creator software GitHub revision",
                            "WARNING",
                            f"Could not verify revision due to network error: {exc}",
                        )
                    except RuntimeError as exc:
                        report.add(
                            SECTION_ID,
                            "Creator software GitHub revision",
                            "WARNING",
                            f"GitHub revision check returned unexpected response: {exc}",
                        )
                else:
                    report.add(
                        SECTION_ID,
                        "Creator software GitHub repository",
                        "WARNING",
                        "Repository not found on GitHub",
                    )
            except RequestException as exc:
                report.add(
                    SECTION_ID,
                    "Creator software GitHub repository",
                    "WARNING",
                    f"Could not verify repository due to network error: {exc}",
                )
            except RuntimeError as exc:
                report.add(
                    SECTION_ID,
                    "Creator software GitHub repository",
                    "WARNING",
                    f"GitHub repository check returned unexpected response: {exc}",
                )
        except ValueError as exc:
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_created_with'",
                "FAIL",
                f"Value '{created_with}' is not a GitHub URL with an @version suffix: {exc}",
            )

    dataset_version = attrs.get("mlcast_dataset_version")
    if dataset_version is None:
        report.add(
            SECTION_ID,
            "Global attribute 'mlcast_dataset_version'",
            "FAIL",
            "Missing required dataset specification version (semver or calver)",
        )
    else:
        try:
            parsed_version = parse_version(dataset_version.strip())
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_dataset_version'",
                "PASS",
                f"Dataset specification version parsed ({parsed_version})",
            )
        except (InvalidVersion, ValueError, AttributeError) as exc:
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_dataset_version'",
                "FAIL",
                f"Version '{dataset_version}' is not valid semver or calver: {exc}",
            )

    dataset_identifier_format = attrs.get(DATASET_IDENTIFIER_FORMAT_ATTRIBUTE)
    format_ok = True
    if dataset_identifier_format is None:
        identifier_format = DEFAULT_DATASET_IDENTIFIER_FORMAT
        report.add(
            SECTION_ID,
            f"Global attribute '{DATASET_IDENTIFIER_FORMAT_ATTRIBUTE}'",
            "PASS",
            f"Default format used ({identifier_format})",
        )
    else:
        try:
            fields = validate_identifier_format(dataset_identifier_format)
            identifier_format = dataset_identifier_format.strip()
            field_list = ", ".join(fields)
            report.add(
                SECTION_ID,
                f"Global attribute '{DATASET_IDENTIFIER_FORMAT_ATTRIBUTE}'",
                "PASS",
                f"Dataset identifier format accepted ({field_list})",
            )
        except ValueError as exc:
            format_ok = False
            identifier_format = None
            report.add(
                SECTION_ID,
                f"Global attribute '{DATASET_IDENTIFIER_FORMAT_ATTRIBUTE}'",
                "FAIL",
                f"Dataset identifier format is invalid: {exc}",
            )

    dataset_identifier = attrs.get(DATASET_IDENTIFIER_ATTRIBUTE)
    if dataset_identifier is None:
        report.add(
            SECTION_ID,
            f"Global attribute '{DATASET_IDENTIFIER_ATTRIBUTE}'",
            "FAIL",
            "Missing required dataset identifier",
        )
    elif not format_ok or identifier_format is None:
        report.add(
            SECTION_ID,
            f"Global attribute '{DATASET_IDENTIFIER_ATTRIBUTE}'",
            "FAIL",
            "Cannot validate dataset identifier because the format is invalid",
        )
    else:
        try:
            parsed_identifier = parse_dataset_identifier(
                dataset_identifier, identifier_format
            )
            detail = ", ".join(
                f"{key}={value}" for key, value in parsed_identifier.items()
            )
            report.add(
                SECTION_ID,
                f"Global attribute '{DATASET_IDENTIFIER_ATTRIBUTE}'",
                "PASS",
                f"Dataset identifier parsed ({detail})",
            )
        except ValueError as exc:
            report.add(
                SECTION_ID,
                f"Global attribute '{DATASET_IDENTIFIER_ATTRIBUTE}'",
                "FAIL",
                f"Dataset identifier '{dataset_identifier}' is invalid: {exc}",
            )

    return report
