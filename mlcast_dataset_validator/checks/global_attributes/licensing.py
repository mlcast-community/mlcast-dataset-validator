from difflib import get_close_matches
from typing import Iterable

import xarray as xr
from license_expression import ExpressionError, get_spdx_licensing

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.2"
LICENSING = get_spdx_licensing()
_KNOWN_LICENSE_KEYS = {key.upper() for key in LICENSING.known_symbols}


def _normalize_spdx(values: Iterable[str]) -> tuple[set[str], list[tuple[str, str]]]:
    """
    Normalize one or more license expressions to canonical SPDX form.

    Returns (normalized_values, errors) where errors is a list of (value, message).
    """
    normalized_values: set[str] = set()
    errors: list[tuple[str, str]] = []

    for value in values:
        try:
            expression = LICENSING.parse(value, validate=True)
        except ExpressionError as exc:
            errors.append((value, str(exc)))
            continue
        normalized_values.add(expression.render())

    return normalized_values, errors


def _suggest_spdx(
    value: str, max_suggestions: int = 3, cutoff: float = 0.6
) -> list[str]:
    """
    Suggest SPDX license identifiers that closely match the provided value.
    """
    return get_close_matches(
        value.upper(), _KNOWN_LICENSE_KEYS, n=max_suggestions, cutoff=cutoff
    )


@log_function_call
def check_license(
    ds: xr.Dataset,
    require_spdx: bool,
    recommended: list[str],
    warn_on_restricted: list[str],
) -> ValidationReport:
    """
    Validate the licensing information in the dataset.

    Parameters:
        ds (xr.Dataset): The dataset to validate.
        require_spdx (bool): Whether a valid SPDX identifier is required.
        recommended (list[str]): List of recommended licenses.
        warn_on_restricted (list[str]): List of restricted license fragments (e.g., "NC", "ND")
            that should generate warnings when present in the license string.

    Returns:
        ValidationReport: A report containing the results of the licensing validation checks.
    """
    report = ValidationReport()

    if "license" not in ds.attrs:
        report.add(
            SECTION_ID,
            "License metadata",
            "FAIL",
            "Missing required 'license' global attribute",
        )
        return report

    license_id = ds.attrs["license"].strip()
    normalized_license_set, errors = _normalize_spdx([license_id])
    normalized_license = next(iter(normalized_license_set), None)
    error = errors[0][1] if errors else None

    from loguru import logger

    logger.debug(
        f"License found: '{license_id}', normalized: '{normalized_license}', error: '{error}'"
    )

    if error:
        suggestions = _suggest_spdx(license_id)
        status = "FAIL" if require_spdx else "WARNING"
        report.add(
            SECTION_ID,
            "License compliance",
            status,
            (
                f"License '{license_id}' is not a valid SPDX expression: {error}"
                + (f". Did you mean: {', '.join(suggestions)}?" if suggestions else "")
            ),
        )
        return report

    normalized_recommended, recommended_errors = _normalize_spdx(recommended)
    restricted_tokens = {token.upper() for token in warn_on_restricted}

    if recommended_errors:
        for value, msg in recommended_errors:
            suggestions = _suggest_spdx(value)
            report.add(
                SECTION_ID,
                "License compliance",
                "WARNING",
                (
                    f"Recommended license '{value}' is not valid SPDX: {msg}"
                    + (
                        f". Did you mean: {', '.join(suggestions)}?"
                        if suggestions
                        else ""
                    )
                ),
            )

    is_recommended = normalized_license in normalized_recommended
    is_restricted = any(
        token in normalized_license.upper() for token in restricted_tokens
    )

    if is_recommended:
        report.add(
            SECTION_ID,
            "License compliance",
            "PASS",
            f"License '{normalized_license}' is recommended and accepted",
        )
    elif is_restricted:
        report.add(
            SECTION_ID,
            "License compliance",
            "WARNING",
            f"License '{normalized_license}' has restrictions (NC/ND)",
        )
    else:
        report.add(
            SECTION_ID,
            "License compliance",
            "WARNING",
            f"License '{normalized_license}' requires case-by-case review",
        )

    return report
