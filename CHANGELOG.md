# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/mlcast-community/mlcast-dataset-validator)

### Added

- Check the CF attributes of the spatial coordinates: `standard_name`, `units` and `axis` on the projected x/y coordinates, `standard_name` and `units` on latitude/longitude (§3.1). Previously a coordinate was accepted by its bare name, so the spec text was not enforced [\#41](https://github.com/mlcast-community/mlcast-dataset-validator/pull/41), @franchg
- Check the CF grid mapping attributes of the crs variable (`grid_mapping_name` and the projection parameters, interpreted with pyproj without falling back to the WKT) and require both the CF attributes and `crs_wkt` to reproduce the stored latitude/longitude at sampled grid points (§4.5). Closes [\#26](https://github.com/mlcast-community/mlcast-dataset-validator/issues/26) [\#41](https://github.com/mlcast-community/mlcast-dataset-validator/pull/41), @franchg
- `pyproj` is now a core dependency [\#41](https://github.com/mlcast-community/mlcast-dataset-validator/pull/41), @franchg
- Require the global attribute `mlcast_dataset_validator_version`, giving the `mlcast-dataset-validator` version that the dataset conforms to (§5.4). A missing or invalid version fails, and a version other than the one running the validation gives a warning. This lets the `mlcast-datasets` catalog check that each dataset matches the validator version its catalog entry is pinned to [\#43](https://github.com/mlcast-community/mlcast-dataset-validator/pull/43), @leifdenby

### Changed

- Radar precipitation spec version bumped to 0.3.0 (new MUST requirements in §3.1 and §4.5) [\#41](https://github.com/mlcast-community/mlcast-dataset-validator/pull/41), @franchg
- Expand the scope of the radar precipitation spec to include single-radar products (previously restricted to multi-radar composites). Single-radar datasets are now in scope provided the valid sensing area supports at least one 256×256 crop at ≤1 km resolution (§3.2), @franchg
- Include an explicit rel_tolerance as a parameter to check_spatial_requirements function and set it to 1% in radar_precipitation.py. This allows a small tolerance for datasets that are marginally above the strict 1.0km threshold, while maintaining the original resolution constraint intent. @jaimecasari

### Fixed

- The chunking check now also requires each chunk to cover the full spatial domain, as §4.1 already specified (1 × height × width). Previously only the time chunk size was checked [\#44](https://github.com/mlcast-community/mlcast-dataset-validator/pull/44), @franchg

### Maintenance

- Add CI check that fails pull requests which don't update `CHANGELOG.md`; PRs that don't need an entry can opt out with the `no changelog` label [\#42](https://github.com/mlcast-community/mlcast-dataset-validator/pull/42), @leifdenby

## [v0.3.0](https://github.com/mlcast-community/mlcast-dataset-validator/releases/tag/v0.3.0)

### Fixed

- Make checks on max spatial resolution (1km) more lenient using math.isclose [\#30](https://github.com/mlcast-community/mlcast-dataset-validator/pull/30), @ladc
- Detect Zarr v3 format from store files (`zarr.json`) instead of relying on `getattr(ds, "zarr_format", 2)` which always defaulted to v2, causing v3 stores to incorrectly fail the consolidated metadata check [\#27](https://github.com/mlcast-community/mlcast-dataset-validator/pull/27), @franchg
- Fix for package version in ci build of html render of specs [\#25](https://github.com/mlcast-community/mlcast-dataset-validator/pull/25), @leifdenby
- Ensure zarr format checks fail if requirements cannot be validated due to missing access to underlying zarr store [\#31](https://github.com/mlcast-community/mlcast-dataset-validator/pull/31), @leifdenby

### Maintenance

- Add info on installing `mlcast-dataset-validator` from PyPI and running from command-line with `uvx` in README [\#32](https://github.com/mlcast-community/mlcast-dataset-validator/pull/32), @leifdenby

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
