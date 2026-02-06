# mlcast-dataset-validator

[![Spec Docs](https://img.shields.io/badge/spec%20docs-HTML-blue)](https://mlcast-community.github.io/mlcast-dataset-validator/)

Dataset validator for the MLCast Intake catalog ([mlcast-datasets](https://github.com/mlcast-community/mlcast-datasets)).

## What is this?

This repository contains the validation tool for datasets contributed to the [MLCast community](https://github.com/mlcast-community) (currently only radar precipitation source datasets). The validator ensures that the datasets meet the technical requirements for inclusion in the MLCast data collection.

### Background

During the MLCast community meeting, multiple entities offered to contribute datasets. To streamline the contribution process and ensure data quality, we developed this validator to help data providers verify that their Zarr archives are compliant with MLCast requirements before submission.

This tool addresses two key needs identified in the community:
1. **Specification compliance** ([#6](https://github.com/mlcast-community/mlcast-datasets/issues/6)): Validates datasets against the formal MLCast Zarr format specification v1.0 (RFC 2119 keywords)
2. **Tool compatibility** ([#5](https://github.com/mlcast-community/mlcast-datasets/issues/5)): Tests that datasets work correctly with common geospatial tools (xarray, GDAL, cartopy)

### What does it validate?

The validator checks both **specification compliance** and **practical tool compatibility**, for example for [radar precipitation datasets it checks](mlcast_dataset_validator/specs/source_data/radar_precipitation.py):

- **Minimum Requirements for Dataset Acceptance:**
    - 2D radar composite at 1km resolution or finer
    - At least 256×256 pixel valid sensing area
    - Minimum 3 years of temporal coverage
    - Consistent spatial domain across all timesteps
    - Data variable in mm (depth), mm/h (rate), or dBZ (reflectivity)

- **Technical Requirements:**
    - GeoZarr format (Zarr v2/v3 with proper georeferencing)
    - CF-compliant coordinate and variable names
    - Correct dimension ordering (time × H × W)
    - Proper chunking strategy (1 chunk per timestep)
    - ZSTD compression (recommended)
    - NaN values for missing/out-of-range data
    - License metadata (CC-BY, CC-BY-SA, OGL, etc.)

- **Tool Compatibility:**
    - xarray can load and slice the data correctly
    - GDAL can interpret the georeferencing (WKT parsing)
    - cartopy can create CRS objects and transform coordinates
    - Cross-tool CRS consistency checks

## How is the tool implemented?

1. **Spec modules are organized by data stage/product**
   Each validator lives under `mlcast_dataset_validator/specs/<data_stage>/<product>.py`. For example, the source data radar precipitation spec is found at `specs/source_data/radar_precipitation.py`, while `specs/training_data/` is prepared for future ML training datasets derived from source data.

2. **Spec sections mirror an `xr.Dataset`**
   Within each spec module, the validation flow follows the dataset layout (coordinates, variables, global attrs, tool compatibility). This makes it easy to place new checks in the appropriate section as the spec evolves.

3. **Inline spec text drives each requirement**
   Every section block contains the human-readable spec text (RFC 2119 wording) followed immediately by function calls that implement the corresponding checks (e.g., `check_coordinate_names`, `check_georeferencing`, `check_gdal_compatibility`). This keeps the specification and enforcement side-by-side.

4. **Checking functions live under `mlcast_dataset_validator/checks/<dataset_section>/<dataset_aspect>.py`**
   Reusable validators for coordinates, data variables, global attrs, and tool compatibility live under paths like `mlcast_dataset_validator/checks/<dataset_section>/<dataset_aspect>.py:check_<dataset_property>`. Specs import the relevant function(s) for each section.

```
mlcast_dataset_validator/
├── specs/
│   ├── source_data/
│   │   └── radar_precipitation.py
│   ├── training_data/
│   │   └── ... (no specs yet)
│   └── cli.py
└── checks/
    ├── coords/
    │   ├── names.py (check_coordinate_names)
    │   ├── spatial.py
    │   ├── temporal.py
    │   └── variable_timestep.py
    ├── data_vars/
    ├── global_attributes/
    └── tool_compatibility/
```


## Example usage

Until `mllam-dataset-validator` is published to PyPI, the easiest way to run it is to use [uv](https://docs.astral.sh/uv/getting-started/installation/) to run it directly from the GitHub repository:

```bash
uvx --with "git+https://github.com/mlcast-community/mlcast-dataset-validator" mlcast.validate_dataset <data_stage> <product> <dataset-path>
```

I.e. you can validate a local Zarr dataset like this:
```bash
uvx --with "git+https://github.com/mlcast-community/mlcast-dataset-validator" mlcast.validate_dataset source_data radar_precipitation /path/to/radar_precip_source.zarr
```

The validator supports also remote zarr hosted in S3 buckets at custom endpoints. We can run it on the Radklim Zarr already available in the intake catalog:

```bash
uvx --with "git+https://github.com/mlcast-community/mlcast-dataset-validator" mlcast.validate_dataset source_data radar_precipitation s3://mlcast-source-datasets/radklim/v0.1.0/5_minutes.zarr/ --s3-endpoint-url https://object-store.os-api.cci2.ecmwf.int --s3-anon
```

Or you can of course clone the repository and run it directly:

```bash
git clone
cd mlcast-sourcedata-validator
pip install -e .
mlcast.validate_dataset source_data radar_precipitation /path/to/zarr/file.zarr
```
