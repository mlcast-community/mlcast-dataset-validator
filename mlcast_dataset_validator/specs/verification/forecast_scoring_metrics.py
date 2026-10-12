#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MLCast specification for forecast scoring metrics
datasets and provides a validation function to check datasets against this
specification. The specification text is written inline with the calls to
checking operations that match the specification requirements.

The checks reused here were written for (and are numbered by) other
specifications, so their results are relabelled with `in_section(...)` to
match the section numbering of this specification.
"""

import textwrap

import xarray as xr
from loguru import logger

from ...checks.coords.dimensions import (
    check_dimension_coordinate_types,
    check_required_dimensions,
)
from ...checks.coords.temporal import check_forecast_case_coverage, check_lead_time
from ...checks.data_vars.data_structure import check_variable_dims_and_dtype
from ...checks.data_vars.data_variable import check_variable_attributes
from ...checks.data_vars.metric_polarity import (
    KNOWN_METRIC_POLARITY,
    POLARITIES,
    check_metric_polarity,
)
from ...checks.data_vars.naming import check_metric_variable_naming
from ...checks.global_attributes.licensing import check_license
from ...checks.global_attributes.mlcast_metadata import check_mlcast_metadata
from ...checks.global_attributes.recommended import check_global_attributes
from ...checks.global_attributes.zarr_format import check_zarr_format
from ..reporting import ValidationReport

VERSION = "0.1.0"
IDENTIFIER = __spec__.name.split(".")[-1]

MLWP_SCORECARD_COMMIT = "2680486"
REQUIRED_DIMS = ("forecast_source", "init_time", "lead_time")
RESERVED_DIMS = ("metric", "variable")


# -------------------------
# Core public API
# -------------------------
def validate_dataset(ds: xr.Dataset) -> ValidationReport:
    """Validate a forecast scoring metrics dataset against the MLCast specification."""
    report = ValidationReport()
    known_metrics = ", ".join(f"`{m}`" for m in KNOWN_METRIC_POLARITY)
    polarities = ", ".join(f"`{p}`" for p in POLARITIES)
    spec_text = f"""
    ---
    data_stage: verification
    product: {IDENTIFIER}
    version: {VERSION}
    ---

    ## 1. Introduction

    This document specifies the requirements for forecast scoring metrics
    datasets, i.e. datasets of metrics applied to score forecasts against a
    truth (such as RMSE, CSI or FSS), to be included in the MLCast data
    collection.
    The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
    "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be
    interpreted as described in RFC 2119.

    A forecast scoring metrics dataset holds one data variable per (metric, physical
    variable) pair. Each element is the score of one forecast, from one forecast
    source, at one lead time (and optionally against one truth source, over one
    spatial region, at one level or threshold), i.e. after the reduction over
    space within a forecast case but *before* the reduction over forecast cases.
    Keeping the scores per forecast case means that any aggregation (means over
    cases, paired differences between forecast sources, confidence intervals and
    significance tests by resampling the cases, seasonal subsets, etc.) can be
    computed from the dataset without access to the forecasts themselves.

    ## 2. Scope

    This specification applies to forecast scoring metrics computed for any
    forecast product (e.g. precipitation nowcasts, numerical or machine-learned
    weather forecasts) against any truth source (e.g. radar observations,
    analyses). It does not constrain how the metrics are computed.

    The layout follows the `{{metric}}.{{variable}}` naming convention of
    [WeatherBench-X](https://github.com/google-research/weatherbenchX), and a
    dataset conforming to this specification can be used directly as input to
    [mlwp-scorecard](https://github.com/mlwp-tools/mlwp-scorecard) (as of
    commit `{MLWP_SCORECARD_COMMIT}`) to produce scorecards.

    (see inline comments below for rest of specification)
    """

    logger.info(str(ds))

    spec_text += """
    ## 3. Dimension and Coordinate Requirements
    """

    spec_text += """
    ### 3.1 Required and Reserved Dimensions

    - The dataset MUST have the dimensions `forecast_source` (the forecast
      sources being verified, e.g. models or model versions), `init_time` (the
      forecast cases, i.e. initialisation/reference times) and `lead_time`.
    - The dimensions `metric` and `variable` MUST NOT be used, these are
      encoded in the data variable names (§4.1).
    - Any other dimension (e.g. `truth_source`, `spatial_region`, `level`,
      `threshold`, `season`) MAY be used.
    - Every dimension MUST have a coordinate variable of the same name.
    """
    report += check_required_dimensions(
        ds,
        required=REQUIRED_DIMS,
        forbidden=RESERVED_DIMS,
    ).in_section("3.1")

    spec_text += """
    ### 3.2 Coordinate Types

    - The values of every dimension coordinate MUST be unique.
    - `forecast_source` MUST hold strings naming the forecast sources.
    - `init_time` MUST be of type `datetime64`.
    - `lead_time` MUST be of type `timedelta64`.
    - If present, `truth_source` and `spatial_region` MUST hold strings.
    - If present, `level` and `threshold` MUST be numeric and SHOULD carry a
      `units` attribute.
    """
    report += check_dimension_coordinate_types(
        ds,
        kinds=dict(
            forecast_source="string",
            init_time="datetime",
            lead_time="timedelta",
            truth_source="string",
            spatial_region="string",
            level="numeric",
            threshold="numeric",
        ),
        recommended_attrs=dict(level=["units"], threshold=["units"]),
    ).in_section("3.2")

    spec_text += """
    ### 3.3 Forecast Cases and Lead Times

    - `init_time` SHOULD be monotonically increasing.
    - The forecast cases SHOULD be numerous enough to resample them in at least
      4 blocks of 10 days, i.e. at least 4 × ⌈10 days / median `init_time`
      spacing⌉ cases (e.g. 40 daily or 80 twice-daily forecasts). This is the
      minimum for the moving-block bootstrap used by mlwp-scorecard by default.
    - All `lead_time` values MUST be non-negative.
    """
    report += check_forecast_case_coverage(
        ds, dim="init_time", min_blocks=4, block_days=10
    ).in_section("3.3")
    report += check_lead_time(ds, dim="lead_time").in_section("3.3")

    spec_text += """
    ## 4. Data Variable Requirements
    """

    spec_text += """
    ### 4.1 Naming

    - Every data variable MUST be named `{metric}.{variable}`, where `{metric}`
      is the name of the verification metric (e.g. `rmse`, `csi`) and
      `{variable}` the name of the verified physical variable (e.g. `2t`, `tp`).
      The name is split on the *last* `.`, so the metric name MAY itself contain
      `.` (e.g. `seeps.v1.5.tp`), and both parts MUST be non-empty.
    - Metric names SHOULD be lower case.
    - The dataset MUST NOT contain data variables that are not scores, such as
      the number of verified forecasts. These are derived by counting the
      finite values (§4.3).
    """
    report += check_metric_variable_naming(ds, separator=".").in_section("4.1")

    spec_text += f"""
    ### 4.2 Metric Polarity

    Whether a higher or lower value of a metric indicates a better forecast
    (the metric's *polarity*) MUST be determinable for every metric:

    - The polarity of the following metrics is known and need not be declared:
      {known_metrics}.
    - Every other metric MUST declare its polarity in the data variable
      attribute `metric_polarity`, with one of the values {polarities}
      (`neutral` for metrics where neither direction is better, e.g. bias, and
      `activity` for measures of variability, e.g. spread).
    - If a known metric declares `metric_polarity`, it MUST agree with the known
      polarity.
    """
    report += check_metric_polarity(
        ds,
        known=KNOWN_METRIC_POLARITY,
        attr="metric_polarity",
        separator=".",
    ).in_section("4.2")

    spec_text += """
    ### 4.3 Data Structure

    - Every data variable MUST have the dimensions `forecast_source`,
      `init_time` and `lead_time`, in any order.
    - A data variable MAY omit other dimensions that do not apply to it (e.g.
      a surface variable has no `level`).
    - Data variables MUST be of floating point type. Missing values (a forecast
      case that was not verified) MUST be encoded as NaN.
    - Every forecast source SHOULD have at least one finite value in every data
      variable.
    """
    report += check_variable_dims_and_dtype(
        ds,
        required_dims=REQUIRED_DIMS,
        dtype_kinds="f",
        require_finite_along="forecast_source",
    ).in_section("4.3")

    spec_text += """
    ### 4.4 Data Variable Attributes

    - Every data variable MUST carry a `units` attribute (`"1"` for
      dimensionless metrics). Since the metric is part of the variable name,
      units can differ between metrics of the same variable (e.g. `rmse.2t` in
      `K` while `acc.2t` is dimensionless).
    - Every data variable SHOULD carry a `long_name` attribute.
    - Data variables MAY carry a CF `standard_name` attribute for the verified
      physical variable.
    """
    report += check_variable_attributes(
        ds,
        required=["units"],
        recommended=["long_name"],
    ).in_section("4.4")

    spec_text += """
    ## 5. Global Attribute Requirements
    """

    spec_text += """
    ### 5.1 Provenance

    The dataset SHOULD include the following global attributes:

    - `title`: A short description of the dataset.
    - `forecasts`: A description of the forecast sources verified.
    - `truth`: A description of the truth source(s) the forecasts are verified
      against.
    """
    report += check_global_attributes(
        ds,
        recommended=["title", "forecasts", "truth"],
    ).in_section("5.1")

    spec_text += """
    ### 5.2 Licensing Requirements

    - The dataset MUST include a global `license` attribute containing a valid SPDX identifier.
    - The following licenses are RECOMMENDED: `CC-BY`, `CC-BY-SA`, `OGL`.
    - Licenses with `NC` or `ND` restrictions SHOULD generate warnings but MAY be accepted after review.
    """
    report += check_license(
        ds,
        require_spdx=True,
        recommended=[
            "CC-BY-SA-2.5",
            "CC-BY-SA-3.0",
            "CC-BY-SA-4.0",
            "CC-BY-2.5",
            "CC-BY-3.0",
            "CC-BY-4.0",
            "OGL-UK-1.0",
            "OGL-UK-2.0",
            "OGL-UK-3.0",
            "OGL-Canada-2.0",
            "GPL-1.0",
            "GPL",
        ],
        warn_on_restricted=["NC", "ND"],
    ).in_section("5.2")

    spec_text += """
    ### 5.3 Storage Format

    - The dataset MUST be stored either as Zarr (version 2 or version 3) or as
      a netCDF file (with suffix `.nc`, `.nc4` or `.cdf`).
    - If Zarr version 2 is used, the dataset MUST include consolidated metadata.
    """
    report += check_zarr_format(
        ds,
        allowed_versions=[2, 3],
        require_consolidated_if_v2=True,
        allow_netcdf=True,
    ).in_section("5.3")

    spec_text += """
    ### 5.4 MLCast Metadata

    The dataset MUST include the following global attributes:

    - `mlcast_created_on`: ISO formatted datetime of dataset creation.
    - `mlcast_dataset_version`: Dataset specification version (semver or calver).
    - `mlcast_dataset_validator_version`: Version of `mlcast-dataset-validator` that the dataset conforms to (e.g., `0.4.0`). It SHOULD match the version of the validator being run.

    Forecast scoring metrics datasets are usually written by a verification tool rather than built by a dedicated dataset repository, so creator and software provenance are optional. The dataset MAY include the following global attributes, which MUST be valid when present:

    - `mlcast_created_by`: Creator contact in `Name <email>` format.
    - `mlcast_created_with`: GitHub URL of the software that created the dataset, including version (e.g., https://github.com/leifdenby/mlcast-verification-tools@v0.1.0). Any GitHub repository is accepted, and the repository/revision MUST exist.
    """
    report += check_mlcast_metadata(
        ds,
        require_identifier=False,
        require_created_by=False,
        require_created_with=False,
        require_dataset_repo_naming=False,
    ).in_section("5.4")

    return report, textwrap.dedent(spec_text)
