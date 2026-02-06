# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.2.0](https://github.com/mlcast-community/mlcast-dataset-validator/releases/tag/v0.2.0)

This release makes the validator easier to use from python and the specs defined in the validator easier to access. This done by allowing for direct calls to validation functions with `xr.Dataset` input. And introducing a cli arg to print selected spec to terminal and adding CI rendering of specs to HTML that are deployted to GitHub Pages for linkable, readable spec docs.

### Added

- Add `--print-spec-markdown` to skip validation/dataset loading, stub all checks, print the selected spec as Markdown, and include YAML front matter (`data_stage`, `product`, `version`) for metadata consumers. [\#17](https://github.com/mlcast-community/mlcast-dataset-validator/pull/17), @leifdenby
- Add CI rendering of specs to HTML and deploy to GitHub Pages for linkable, readable spec docs. [\#22](https://github.com/mlcast-community/mlcast-dataset-validator/pull/22), [\#23](https://github.com/mlcast-community/mlcast-dataset-validator/pull/23), @leifdenby
- Make `mlcast_dataset_validator.specs.source_data.radar_precipitation.validate_dataset()` accept `xr.Dataset` input so it can be called from Python CI in mlcast-community/mlcast-datasets, and expose a report table print target to allow stdout output. [mlcast-datasets\#22](https://github.com/mlcast-community/mlcast-datasets/pull/22)

## [v0.1.0](https://github.com/mlcast-community/mlcast-dataset-validator/releases/tag/v0.1.0)

First release of validator for MLCast datasets that enforces spec compliance and practical tool compatibility. This first release focuses on core functionality and implements a first revision of the dataset specification for radar precipitation datasets.

What’s included:

- Spec compliance checks:
  - CF‑compliant coordinate/variable naming and metadata
  - Spatial resolution/domain constraints and stable grid
  - Temporal coverage, monotonic time, variable timestep handling with consistent_timestep_start
  - Explicit missing_times support for inferred gaps
  - Dimension order, chunking strategy, compression recommendations
  - GeoZarr georeferencing and CRS metadata requirements
  - Global attributes and licensing checks
- Tool compatibility checks:
  - xarray load/slice
  - GDAL CRS parsing
  - cartopy CRS object creation and transformations
- CLI:
  - `mlcast.validate_dataset <stage> <product> <path>`
  - Works with local Zarr or remote S3 (custom endpoints, anonymous access)
