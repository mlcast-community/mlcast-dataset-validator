#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MLCast specification for 2D radar precipitation
composite datasets and provides a validation function to check datasets
against this specification. The specification text is written inline with the
calls to checking operations that match the specification requirements.
"""

import xarray as xr
from loguru import logger

from ...checks.coords.names import check_coordinate_names
from ...checks.coords.spatial import check_spatial_requirements
from ...checks.coords.temporal import check_temporal_requirements
from ...checks.data_vars import naming
from ...checks.data_vars.chunking import check_chunking_strategy
from ...checks.data_vars.compression import check_compression
from ...checks.data_vars.data_structure import check_data_structure
from ...checks.data_vars.georeferencing import check_georeferencing
from ...checks.global_attributes.conditional import check_conditional_global_attributes
from ...checks.global_attributes.licensing import check_license
from ...checks.global_attributes.mlcast_metadata import check_mlcast_metadata
from ...checks.global_attributes.zarr_format import check_zarr_format
from ...checks.tool_compatibility.cartopy import check_cartopy_compatibility
from ...checks.tool_compatibility.gdal import check_gdal_compatibility
from ..base import ValidationReport

VERSION = "0.2.0"
IDENTIFIER = __spec__.name.split(".")[-1]


# -------------------------
# Core public API
# -------------------------
def validate_dataset(ds: xr.Dataset) -> ValidationReport:
    """Validate a radar precipitation dataset against the MLCast specification."""
    report = ValidationReport()
    spec_text = """
    ## 1. Introduction

    This document specifies the requirements for 2D radar precipitation and
    reflectivity composite datasets to be included in the MLCast data collection.
    The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
    "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be
    interpreted as described in RFC 2119.

    ## 2. Scope

    This specification applies to 2D radar composite datasets (merged from multiple
    radar sources) intended for machine learning applications in weather and
    climate research. Single-radar datasets are explicitly excluded from this
    specification.

    (see inline comments below for rest of specification)
    """

    logger.info("Validating provided dataset.")
    logger.info(str(ds))

    spec_text += """
    ## 3. Coordinate Requirements
    """

    spec_text += """
    ### 3.1 Coordinate Variables

    > "The dataset MUST expose CF-compliant coordinates: latitude/longitude and projected x/y."
    > "Coordinate metadata MUST provide `standard_name`/`axis`/`units` per CF (with a valid `time` coordinate as well)."
    """
    report += check_coordinate_names(
        ds,
        require_time_coord=True,
        require_projected_coords=True,
        require_latlon_coords=True,
    )

    spec_text += """
    ### 3.2 Spatial Requirements

    > "The dataset MUST provide 2D radar composites with a spatial resolution of 1 kilometer or finer."
    > "The valid sensing area MUST support at least one 256×256 pixel square crop that is fully contained within the radar sensing range."
    > "The spatial domain, including resolution, size, and geographical coverage, MUST remain constant across all times in the archive."
    """
    report += check_spatial_requirements(
        ds,
        max_resolution_km=1.0,
        min_crop_size=(256, 256),
        require_constant_domain=True,
    )

    spec_text += """
    ### 3.3 Temporal Requirements

    - "The timestep (the duration between successive time values for which a single data-point is valid) MAY be variable throughout the archive, but in that case a global attribute named `consistent_timestep_start` MUST be included to indicate the first timestamp where regular timestepping begins. In the absence of this attribute, the timestep MUST be regular throughout the archive."
    - "Times for which data is missing MUST be given expicitly in the variable `missing_times` as CF-compliant time values. The timestep is defined as the interval between consecutive times (including missing times). These times MUST NOT be included in the main time coordinate."
    - "Time values MUST be strictly monotonically increasing."
    """
    report += check_temporal_requirements(
        ds,
        min_years=3,
        allow_variable_timestep=True,
    )

    spec_text += """
    ## 4. Data Variable Requirements
    """

    spec_text += """
    ### 4.1 Chunking Strategy

    > "The dataset MUST use a chunking strategy of 1 × height × width (one chunk per timestep)."
    """
    report += check_chunking_strategy(
        ds,
        time_chunksize=1,
    )

    spec_text += """
    ### 4.2 Compression

    > "The main data arrays MUST use compression to reduce storage requirements."
    > "ZSTD compression is RECOMMENDED for optimal performance of the main data arrays."
    > "Coordinate arrays MAY use different compression algorithms (e.g., lz4) as appropriate."
    """
    report += check_compression(
        ds,
        require_compression=True,
        recommended_compression="zstd",
        allow_coord_algs=["lz4"],
    )

    spec_text += """
    ### 4.3 Data Structure

    > "The main data variable MUST be encoded with dimensions in the order: time × height (y, lat) × width (x, lon)."
    > "The data type MUST be floating-point (float16, float32, or float64)."
    """
    report += check_data_structure(
        ds,
        dim_order=("time", "y", "x"),
        allowed_dtypes=["float16", "float32", "float64"],
    )

    spec_text += """
    ### 4.4 Data Variable Naming and Attributes

    > "The data variable name SHOULD be a CF convention standard name or use a sensible name from the ECMWF parameter database."
    > "The data variable MUST include the `long_name`, `standard_name` and `units` attributes following CF conventions."
    """
    allowed_standard_names = (
        "rainfall_flux",
        "precipitation_flux",
        "equivalent_reflectivity_factor",
        "precipitation_amount",
        "rainfall_amount",
    )
    report += naming.check_names_and_attrs(
        ds,
        allowed_standard_names=allowed_standard_names,
    )

    spec_text += """
    ### 4.5 Georeferencing

    > "The dataset MUST include proper georeferencing information following the GeoZarr specification."
    > "The data variable MUST include a `grid_mapping` attribute that references the coordinate reference system (crs) variable."
    > "The crs variable MUST include both a `spatial_ref` and a `crs_wkt` attribute with a WKT string."
    """
    report += check_georeferencing(
        ds,
        require_geozarr=True,
        require_grid_mapping=True,
        crs_attrs=["spatial_ref", "crs_wkt"],
        require_bbox=True,
    )

    spec_text += """
    ## 5. Global Attribute Requirements
    """

    spec_text += """
    ### 5.1 Conditional Global Attributes

    > "The global attribute `consistent_timestep_start` is CONDITIONALLY REQUIRED if the dataset uses a variable timestep. It MUST be an ISO formatted datetime string indicating the first timestamp where regular timestepping begins."
    """
    report += check_conditional_global_attributes(
        ds,
        conditional_attrs=["consistent_timestep_start"],
    )

    spec_text += """
    ### 5.2 Licensing Requirements

    > "The dataset MUST include a global `license` attribute containing a valid SPDX identifier."
    > "The following licenses are RECOMMENDED: `CC-BY`, `CC-BY-SA`, `OGL`."
    > "Licenses with `NC` or `ND` restrictions SHOULD generate warnings but MAY be accepted after review."
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
    )

    spec_text += """
    ### 5.3 Zarr Format

    > "The dataset MUST use Zarr version 2 or version 3 format."
    > "If Zarr version 2 is used, the dataset MUST include consolidated metadata."
    """
    report += check_zarr_format(
        ds,
        allowed_versions=[2, 3],
        require_consolidated_if_v2=True,
    )

    spec_text += """
    ### 5.4 MLCast Metadata

    > "The dataset MUST include the following global attributes:
    >   - `mlcast_created_on`: ISO formatted datetime of dataset creation.
    >   - `mlcast_created_by`: Creator contact in `Name <email>` format.
    >   - `mlcast_created_with`: GitHub URL of the creating software including version (e.g., https://github.com/mlcast-community/mlcast-dataset-radklim@v0.1.0) and the repository/revision MUST exist.
    >   - `mlcast_dataset_version`: Dataset specification version (semver or calver).
    >   - `mlcast_dataset_identifier`: Unique dataset identifier formatted as `<country_code>-<entity>-<physical_variable>` by default.
    >   - `mlcast_dataset_identifier_format`: OPTIONAL format string that MUST start with `<country_code>-<entity>-<physical_variable>` and MAY include only the approved identifier parts: `country_code`, `entity`, `physical_variable`, `time_resolution`, `common_name`."
    """
    report += check_mlcast_metadata(ds)

    spec_text += """
    ## 6. Tool Compatibility Requirements

    Practical interoperability checks derived from the standalone validator.
    """

    spec_text += """
    ### 6.1 GDAL Compatibility

    > "The dataset SHOULD expose georeferencing metadata readable by GDAL, including a CRS WKT."
    > "A basic GeoTIFF export SHOULD roundtrip through GDAL with geotransform/projection metadata."
    """
    report += check_gdal_compatibility(ds)

    spec_text += """
    ### 6.2 Cartopy Compatibility

    > "The CRS WKT SHOULD be parseable by cartopy."
    > "Coordinate grids SHOULD transform cleanly into PlateCarree for mapping workflows."
    """
    report += check_cartopy_compatibility(ds)

    return report
