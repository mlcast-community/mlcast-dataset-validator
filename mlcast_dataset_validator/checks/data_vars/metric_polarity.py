from __future__ import annotations

from typing import Dict

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from ..data_vars_filter import iter_data_vars
from . import SECTION_ID as PARENT_SECTION_ID
from .naming import split_metric_variable_name

SECTION_ID = f"{PARENT_SECTION_ID}.6"

POLARITIES = ("lower_is_better", "higher_is_better", "neutral", "activity")

# Metrics whose polarity is known without being declared. Mirrors
# `METRIC_POLARITY` in mlwp-scorecard's `polarity.py` as of commit 2680486
# (https://github.com/mlwp-tools/mlwp-scorecard)
KNOWN_METRIC_POLARITY: Dict[str, str] = {
    "rmse": "lower_is_better",
    "rmsef": "lower_is_better",
    "mae": "lower_is_better",
    "mse": "lower_is_better",
    "bias": "neutral",
    "crps": "lower_is_better",
    "crpss": "higher_is_better",
    "acc": "higher_is_better",
    "csi": "higher_is_better",
    "ets": "higher_is_better",
    "spread": "activity",
    "stdev": "activity",
    "activity": "activity",
}


@log_function_call
def check_metric_polarity(
    ds: xr.Dataset,
    *,
    known: Dict[str, str] = KNOWN_METRIC_POLARITY,
    attr: str = "metric_polarity",
    separator: str = ".",
) -> ValidationReport:
    """
    Check that the polarity (whether higher or lower is better) of every
    metric is either known or declared.

    The metric of each data variable is taken from its `{metric}.{variable}`
    name. A metric not in `known` MUST declare its polarity in the variable
    attribute `attr`, and a declared polarity MUST agree with `known`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose data variables should be evaluated.
    known : Dict[str, str], optional
        Mapping of metric name to polarity for metrics that need no declaration.
    attr : str, optional
        Name of the variable attribute declaring the polarity.
    separator : str, optional
        Separator between metric and variable name.

    Returns
    -------
    ValidationReport
        Report with one result per data variable. Variables whose names can't
        be split are skipped, they are reported by the naming check.
    """
    report = ValidationReport()

    for name, data_array in iter_data_vars(ds):
        try:
            metric, _ = split_metric_variable_name(str(name), separator)
        except ValueError:
            continue

        requirement = f"Metric polarity for '{name}'"
        declared = data_array.attrs.get(attr)
        if declared is not None and declared not in POLARITIES:
            report.add(
                SECTION_ID,
                requirement,
                "FAIL",
                f"Attribute '{attr}={declared}' is not one of {list(POLARITIES)}",
            )
        elif metric in known:
            if declared is None or declared == known[metric]:
                report.add(
                    SECTION_ID,
                    requirement,
                    "PASS",
                    f"Metric '{metric}' is known to be {known[metric]}",
                )
            else:
                report.add(
                    SECTION_ID,
                    requirement,
                    "FAIL",
                    f"Attribute '{attr}={declared}' contradicts the known "
                    f"polarity of '{metric}' ({known[metric]})",
                )
        elif declared is None:
            report.add(
                SECTION_ID,
                requirement,
                "FAIL",
                f"Metric '{metric}' is not one of the known metrics "
                f"{sorted(known)}, so its polarity MUST be declared in the "
                f"variable attribute '{attr}' (one of {list(POLARITIES)})",
            )
        else:
            report.add(
                SECTION_ID,
                requirement,
                "PASS",
                f"Metric '{metric}' is declared {declared}",
            )

    return report
