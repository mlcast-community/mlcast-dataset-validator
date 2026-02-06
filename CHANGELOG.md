# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `--print-spec-markdown` to skip validation/dataset loading, stub all checks, print the selected spec as Markdown, and include YAML front matter (`data_stage`, `product`, `version`) for metadata consumers. [\#17](https://github.com/mlcast-community/mlcast-dataset-validator/pull/17), @leifdenby
- Add CI rendering of specs to HTML and deploy to GitHub Pages for linkable, readable spec docs. [\#22](https://github.com/mlcast-community/mlcast-dataset-validator/pull/22), [\#23](https://github.com/mlcast-community/mlcast-dataset-validator/pull/23), @leifdenby

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
