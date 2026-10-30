#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MLCast specification for raw detector counts from
geostationary satellite imagers, stored together with the calibration
coefficients that convert them to radiances, and provides a validation
function to check datasets against this specification. The specification text
is written inline with the calls to checking operations that match the
specification requirements.
"""

import textwrap
from typing import Optional, Tuple

import xarray as xr
from loguru import logger

from ...checks.coords.channels import check_channel_coordinate
from ...checks.coords.names import check_coordinate_attributes, check_coordinate_names
from ...checks.coords.spatial import check_spatial_requirements
from ...checks.coords.temporal import check_temporal_requirements
from ...checks.data_vars import naming
from ...checks.data_vars.calibration import check_counts_calibration
from ...checks.data_vars.chunking import check_chunking_strategy
from ...checks.data_vars.compression import check_compression
from ...checks.data_vars.data_structure import check_data_structure
from ...checks.data_vars.georeferencing import check_georeferencing
from ...checks.global_attributes.conditional import check_conditional_global_attributes
from ...checks.global_attributes.licensing import check_license
from ...checks.global_attributes.mlcast_metadata import check_mlcast_metadata
from ...checks.global_attributes.platform import check_platform_attributes
from ...checks.global_attributes.zarr_format import check_zarr_format
from ...checks.tool_compatibility.cartopy import check_cartopy_compatibility
from ...checks.tool_compatibility.gdal import check_gdal_compatibility
from ..reporting import ValidationReport

VERSION = "0.1.0"
IDENTIFIER = __spec__.name.split(".")[-1]

SPATIAL_DIMS = ("y", "x")
CHANNEL_DIM = "channel"
GRIDDED_DIM_ORDER = ("time", "y", "x", CHANNEL_DIM)
CALIBRATION_DIM_ORDER = ("time", CHANNEL_DIM)

COUNTS_DTYPES = ("uint8", "uint16", "uint32", "int16", "int32")
FLAG_DTYPES = ("int8", "uint8", "int16", "uint16", "int32", "uint32")
PIXEL_TIME_DTYPES = (
    "datetime64[s]",
    "datetime64[ms]",
    "datetime64[us]",
    "datetime64[ns]",
)
CALIBRATION_DTYPES = ("float32", "float64")

# Roles of data variables (see `checks.data_vars_filter.classify_data_vars`)
# that span the spatial dimensions, and those that are not observations
GRIDDED_ROLES = ("observation", "quality_flag", "pixel_time")
NON_OBSERVATION_ROLES = ("quality_flag", "pixel_time", "calibration", "auxiliary")


# -------------------------
# Core public API
# -------------------------
def validate_dataset(ds: Optional[xr.Dataset]) -> Tuple[ValidationReport, str]:
    """
    Validate a geostationary satellite counts dataset against the MLCast specification.

    Parameters
    ----------
    ds : xr.Dataset or None
        Dataset to validate. ``None`` is used (together with
        ``skip_all_checks``) to only render the specification text.

    Returns
    -------
    Tuple[ValidationReport, str]
        The validation report and the specification text as Markdown.
    """
    report = ValidationReport()
    spec_text = f"""
    ---
    data_stage: source_data
    product: {IDENTIFIER}
    version: {VERSION}
    ---

    ## 1. Introduction

    This document specifies the requirements for datasets of raw detector counts
    from geostationary satellite imagers to be included in the MLCast data
    collection. The counts are stored together with the calibration coefficients
    needed to convert them to radiances.
    The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
    "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be
    interpreted as described in RFC 2119.

    ## 2. Scope

    This specification applies to imagery from geostationary satellite imagers
    (e.g. MTG FCI, MSG SEVIRI, GOES ABI, Himawari AHI) intended for machine
    learning applications in weather and climate research.

    - The data MUST be given on the native fixed grid of the instrument in the
      geostationary projection, without regridding or parallax correction.
    - Each dataset MUST contain only channels that share the same native grid.
      Channels with different native resolutions (e.g. the 1 km and 2 km
      channels of MTG FCI) MUST be stored as separate datasets, one per
      resolution, with the same `time` coordinate.

    (see inline comments below for rest of specification)
    """

    logger.info(str(ds))

    spec_text += """
    ## 3. Coordinate Requirements
    """

    spec_text += """
    ### 3.1 Coordinate Variables

    - The dataset MUST expose CF-compliant coordinates: latitude/longitude and projected x/y, as well as a valid `time` coordinate.
    - The projected coordinates MUST carry the CF attributes `standard_name` (`projection_x_coordinate` / `projection_y_coordinate`), `units` (a length unit, e.g. `m`) and `axis` (`X` / `Y`).
    - The latitude/longitude coordinates MUST carry the CF attributes `standard_name` (`latitude` / `longitude`) and `units` (`degrees_north` / `degrees_east`). Grid points that do not view the Earth (off the disc) MUST have NaN latitude/longitude.
    """
    report += check_coordinate_names(
        ds,
        require_time_coord=True,
        require_projected_coords=True,
        require_latlon_coords=True,
    )
    report += check_coordinate_attributes(
        ds,
        require_projected_attrs=True,
        require_latlon_attrs=True,
    )

    spec_text += """
    ### 3.2 Spatial Requirements

    - The spatial dimensions MUST be named `y` and `x`.
    - The dataset MUST have a spatial resolution (at the sub-satellite point) of 3 kilometers (with 1% tolerance) or finer.
    - The spatial domain MUST support at least one 256×256 pixel square crop.
    - The spatial domain, including resolution, size, and geographical coverage, MUST remain constant across all times in the archive.
    """
    report += check_spatial_requirements(
        ds,
        max_resolution_km=3.0,
        rel_tol=1e-2,
        min_crop_size=(256, 256),
        require_constant_domain=True,
        spatial_dims=SPATIAL_DIMS,
    )

    spec_text += """
    ### 3.3 Temporal Requirements

    - The `time` coordinate gives the nominal start time of each repeat cycle (e.g. each full-disc scan).
    - The timestep MAY be variable throughout the archive, but in that case a global attribute named `consistent_timestep_start` MUST be included to indicate the first timestamp where regular timestepping begins. In the absence of this attribute, the timestep MUST be regular throughout the archive.
    - Times for which data is missing MUST be given explicitly in the variable `missing_times` as CF-compliant time values. These times MUST NOT be included in the main time coordinate, and the time coordinate MUST NOT contain NaT (not-a-time) values.
    - Time values MUST be strictly monotonically increasing.
    - The archive MUST cover at least 1 year.
    """
    report += check_temporal_requirements(
        ds,
        min_years=1,
        allow_variable_timestep=True,
    )

    spec_text += f"""
    ### 3.4 Spectral Channels

    - The spectral channels MUST be stored along a dimension named `{CHANNEL_DIM}`.
    - The `{CHANNEL_DIM}` dimension MUST have a coordinate of unique string channel names (e.g. `ir_105`), so that channels can be selected by name.
    - The `{CHANNEL_DIM}` dimension SHOULD have a coordinate giving the central wavelength of each channel, with `standard_name: radiation_wavelength` and a length unit (e.g. `um`).
    """
    report += check_channel_coordinate(ds, channel_dim=CHANNEL_DIM)

    spec_text += """
    ## 4. Data Variable Requirements
    """

    spec_text += """
    ### 4.1 Chunking Strategy

    - Every time-dependent data variable MUST be chunked with one timestep per chunk.
    - The spatial dimensions MAY be split into several chunks (a full disc at 1 km is too large for a single chunk). Using Zarr v3 sharding with one shard per timestep and channel covering the full spatial domain is RECOMMENDED.
    """
    report += check_chunking_strategy(
        ds,
        time_chunksize=1,
        require_full_spatial_chunks=False,
        time_dependent_only=True,
    )

    spec_text += """
    ### 4.2 Compression

    - The data arrays MUST use compression to reduce storage requirements.
    - ZSTD compression is RECOMMENDED for optimal performance of the data arrays.
    - Coordinate arrays MAY use different compression algorithms (e.g., lz4) as appropriate.
    """
    report += check_compression(
        ds,
        require_compression=True,
        recommended_compression="zstd",
        allow_coord_algs=["lz4"],
    )

    dims = " × ".join(GRIDDED_DIM_ORDER)
    spec_text += f"""
    ### 4.3 Data Structure

    - Counts variables (the variables spanning `y` and `x` that hold the measured counts), quality flag variables and per-pixel time variables MUST be encoded with dimensions in the order: {dims}.
    - Counts variables MUST be stored as integers (e.g. `uint16`) on disk. A `_FillValue` MAY be set to mark missing pixels.
    - Quality flag variables (variables with CF `flag_masks` or `flag_values`) MUST be stored as integers.
    - Per-pixel time variables (e.g. the acquisition time of each pixel) MUST be CF-encoded times that decode to datetime values.
    - Calibration coefficient variables MUST be floating-point (float32 or float64) with dimensions in the order: time × channel.
    """
    report += check_data_structure(
        ds,
        dim_order=GRIDDED_DIM_ORDER,
        allowed_dtypes=COUNTS_DTYPES,
        use_encoded_dtype=True,
        roles=["observation"],
    )
    report += check_data_structure(
        ds,
        dim_order=GRIDDED_DIM_ORDER,
        allowed_dtypes=FLAG_DTYPES,
        use_encoded_dtype=True,
        roles=["quality_flag"],
    )
    report += check_data_structure(
        ds,
        dim_order=GRIDDED_DIM_ORDER,
        allowed_dtypes=PIXEL_TIME_DTYPES,
        roles=["pixel_time"],
    )
    report += check_data_structure(
        ds,
        dim_order=CALIBRATION_DIM_ORDER,
        allowed_dtypes=CALIBRATION_DTYPES,
        roles=["calibration"],
    )

    spec_text += """
    ### 4.4 Data Variable Naming and Attributes

    - Counts variables MUST include the `long_name` and `units` attributes, with `units: 1`. As there is no CF standard name for raw counts, they do not need a `standard_name`.
    - Every other data variable (quality flags, per-pixel times, calibration coefficients, ...) MUST include the `long_name` and `units` attributes. For CF-encoded times the `units` (e.g. `seconds since 2000-01-01`) are those of the time encoding.
    """
    report += naming.check_required_attrs(
        ds, required=("long_name", "units"), roles=["observation"]
    )
    report += naming.check_required_attrs(
        ds, required=("long_name", "units"), roles=NON_OBSERVATION_ROLES
    )

    spec_text += """
    ### 4.5 Georeferencing

    - The dataset MUST include proper georeferencing information following the GeoZarr specification.
    - Every data variable spanning `y` and `x` MUST include a `grid_mapping` attribute that references the coordinate reference system (crs) variable. Data variables without spatial dimensions (e.g. calibration coefficients) MUST NOT be required to have one.
    - The crs variable MUST include both a `spatial_ref` and a `crs_wkt` attribute with a WKT string.
    - The crs variable MUST also carry the CF grid mapping attributes (CF conventions section 5.6 and appendix F) for the `geostationary` grid mapping: `grid_mapping_name`, `perspective_point_height`, `longitude_of_projection_origin`, `sweep_angle_axis` and the ellipsoid parameters, so that the CRS can be reconstructed without parsing the WKT.
    - The CF grid mapping attributes and the `crs_wkt` string MUST both georeference the grid consistently with the latitude/longitude coordinates (within 1e-4 degrees at sampled grid points that view the Earth).
    """
    report += check_georeferencing(
        ds,
        require_geozarr=True,
        require_grid_mapping=True,
        crs_attrs=["spatial_ref", "crs_wkt"],
        require_bbox=True,
        require_cf_grid_mapping=True,
        latlon_tolerance_deg=1e-4,
        roles=GRIDDED_ROLES,
    )

    spec_text += """
    ### 4.6 Calibration

    - Each counts variable MUST list the variables holding its calibration coefficients (e.g. `slope` and `offset`) in its CF `ancillary_variables` attribute, together with any other ancillary variables (e.g. quality flags and per-pixel times).
    - Calibration coefficient variables MUST NOT span the spatial dimensions. They MAY vary along `time` and `channel` only, and MUST include the `long_name` and `units` attributes.
    - Each counts variable SHOULD have a `comment` attribute describing how to convert the counts to radiances with the coefficients (e.g. `radiance = offset + slope * counts`).
    """
    report += check_counts_calibration(
        ds,
        spatial_dims=SPATIAL_DIMS,
        calibration_dims=CALIBRATION_DIM_ORDER,
    )

    spec_text += """
    ## 5. Global Attribute Requirements
    """

    spec_text += """
    ### 5.1 Conditional Global Attributes

    - The global attribute `consistent_timestep_start` is CONDITIONALLY REQUIRED if the dataset uses a variable timestep. It MUST be an ISO formatted datetime string indicating the first timestamp where regular timestepping begins.
    """
    report += check_conditional_global_attributes(
        ds,
        conditional_attrs=["consistent_timestep_start"],
    )

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
        ],
        warn_on_restricted=["NC", "ND"],
    )

    spec_text += """
    ### 5.3 Zarr Format

    - The dataset MUST use Zarr version 2 or version 3 format.
    - If Zarr version 2 is used, the dataset MUST include consolidated metadata.
    """
    report += check_zarr_format(
        ds,
        allowed_versions=[2, 3],
        require_consolidated_if_v2=True,
    )

    spec_text += """
    ### 5.4 MLCast Metadata

    The dataset MUST include the following global attributes:

    - `mlcast_created_on`: ISO formatted datetime of dataset creation.
    - `mlcast_created_by`: Creator contact in `Name <email>` format. Multiple creators MAY be given as a comma-separated list (e.g., `A <a@x.org>, B <b@y.org>`).
    - `mlcast_created_with`: GitHub URL of the creating software including version (e.g., https://github.com/mlcast-community/mlcast-dataset-radklim@v0.1.0) and the repository/revision MUST exist.
    - `mlcast_dataset_version`: Dataset specification version (semver or calver).
    - `mlcast_dataset_validator_version`: Version of `mlcast-dataset-validator` that the dataset conforms to (e.g., `0.4.0`). It SHOULD match the version of the validator being run.
    - `mlcast_dataset_identifier`: Unique dataset identifier formatted as `<country_code>-<entity>-<physical_variable>` by default.
    - `mlcast_dataset_identifier_format`: OPTIONAL format string that MUST start with `<country_code>-<entity>-<physical_variable>` and MAY include only the approved identifier parts: `country_code`, `entity`, `physical_variable`, `time_resolution`, `common_name`.
    """
    report += check_mlcast_metadata(ds)

    spec_text += """
    ### 5.5 Platform and Instrument

    - The dataset MUST include the global attribute `platform` naming the satellite (e.g. `MTG-I1`).
    - The dataset MUST include the global attribute `instrument` naming the imager (e.g. `FCI`).
    """
    report += check_platform_attributes(ds, required=("platform", "instrument"))

    spec_text += """
    ## 6. Tool Compatibility Requirements

    Practical interoperability checks derived from the standalone validator.
    """

    spec_text += """
    ### 6.1 GDAL Compatibility

    - The dataset SHOULD expose georeferencing metadata readable by GDAL, including a CRS WKT.
    - A basic GeoTIFF export of one timestep and channel SHOULD roundtrip through GDAL with geotransform/projection metadata.
    """
    report += check_gdal_compatibility(ds, role="observation")

    spec_text += """
    ### 6.2 Cartopy Compatibility

    - A cartopy projection SHOULD be constructible from the CF grid mapping attributes. Cartopy cannot build a plottable projection from the WKT of a geostationary projection, because its footprint is not a latitude/longitude box.
    - Coordinate grids SHOULD transform cleanly into PlateCarree for mapping workflows (grid points off the disc excepted).
    """
    report += check_cartopy_compatibility(ds, crs_source="cf", role="observation")

    return report, textwrap.dedent(spec_text)
