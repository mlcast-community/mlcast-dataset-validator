"""Generic CLI for running validator specs by data_stage/product."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from typing import Dict, List, Sequence

from loguru import logger

from .. import __version__
from .reporting import skip_all_checks

SPEC_PACKAGE = "mlcast_dataset_validator.specs"


def _discover_catalog() -> Dict[str, List[str]]:
    """
    Discover available data_stage/product combinations under ``specs``.

    Returns
    -------
    dict
        Mapping of data_stage names to sorted lists of product module names.
    """
    catalog: Dict[str, List[str]] = {}
    specs_pkg = importlib.import_module(SPEC_PACKAGE)

    for _, data_stage, is_pkg in pkgutil.iter_modules(specs_pkg.__path__):
        if not is_pkg or data_stage.startswith("_"):
            continue

        data_stage_pkg = importlib.import_module(f"{SPEC_PACKAGE}.{data_stage}")
        product_names: List[str] = []
        for _, product_name, is_subpkg in pkgutil.iter_modules(data_stage_pkg.__path__):
            if is_subpkg or product_name.startswith("_"):
                continue
            product_names.append(product_name)

        if product_names:
            catalog[data_stage] = sorted(product_names)

    return catalog


def _format_catalog(catalog: Dict[str, Sequence[str]]) -> str:
    """Format the data_stage/product catalog for human-readable CLI output."""
    lines = []
    for data_stage in sorted(catalog):
        products = ", ".join(catalog[data_stage])
        lines.append(f"  - {data_stage}: {products}")
    return "\n".join(lines)


def build_parser(catalog: Dict[str, List[str]]) -> argparse.ArgumentParser:
    description = (
        "Run the MLCast dataset validator for a specific data_stage/product combination.\n"
        "Stages correspond to subpackages under `specs` (e.g., `source_data`)."
    )
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data_stage",
        type=str,
        nargs="?",
        help="Validator data_stage (e.g., source_data). Use --list to view options.",
    )
    parser.add_argument(
        "product",
        type=str,
        nargs="?",
        help="Product identifier within the data_stage (e.g., radar_precipitation).",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        nargs="?",
        help="Path or URL to the Zarr dataset to validate.",
    )
    parser.add_argument(
        "--s3-endpoint-url",
        type=str,
        default=None,
        help="Optional S3 endpoint URL for accessing the Zarr dataset.",
    )
    parser.add_argument(
        "--s3-anon",
        action="store_true",
        help="Use anonymous access for S3 storage.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available data_stage/product combinations and exit.",
    )
    parser.add_argument(
        "--print-spec-markdown",
        action="store_true",
        help=(
            "Print the selected specification as Markdown without running validation checks. "
            "Requires data_stage and product, dataset_path is optional."
        ),
    )
    parser.epilog = "Available combinations:\n" + _format_catalog(catalog)
    return parser


def _load_validator_module(data_stage: str, product: str):
    module_name = f"{SPEC_PACKAGE}.{data_stage}.{product}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - argparse ensures help
        raise SystemExit(f"Unknown validator module '{module_name}': {exc}") from exc

    if not hasattr(module, "validate_dataset"):
        raise SystemExit(f"Module '{module_name}' does not expose validate_dataset().")

    return module


@logger.catch
def main(argv: Sequence[str] | None = None) -> int:
    catalog = _discover_catalog()
    parser = build_parser(catalog)
    args = parser.parse_args(argv)

    if args.list:
        print("Implemented specifications:")
        print(_format_catalog(catalog))
        return 0

    if args.data_stage and not args.product:
        if args.data_stage not in catalog:
            parser.error(
                f"Unknown data_stage '{args.data_stage}'. Available data_stages: {', '.join(sorted(catalog))}."
            )
        print(f"Implemented specifications for data_stage '{args.data_stage}':")
        for product in catalog[args.data_stage]:
            print(f"  - {product}")
        return 0

    if args.print_spec_markdown:
        if not (args.data_stage and args.product):
            parser.error(
                "--print-spec-markdown requires a data_stage and product to select the specification."
            )
    elif not (args.data_stage and args.product and args.dataset_path):
        parser.print_help()
        return 1

    data_stage = args.data_stage
    product = args.product

    if data_stage not in catalog:
        parser.error(
            f"Unknown data_stage '{data_stage}'. Available data_stages: {', '.join(sorted(catalog))}."
        )
    if product not in catalog[data_stage]:
        parser.error(
            f"Unknown product '{product}' for data_stage '{data_stage}'. "
            f"Available products: {', '.join(catalog[data_stage])}."
        )

    module = _load_validator_module(data_stage, product)

    storage_options = {}
    if args.s3_endpoint_url:
        storage_options["endpoint_url"] = args.s3_endpoint_url
    if args.s3_anon:
        storage_options["anon"] = True

    logger.info(
        f"Running {data_stage}/{product} validator (mlcast-dataset-validator "
        f"{__version__})"
    )

    if args.print_spec_markdown:
        with skip_all_checks():
            _, spec_text = module.validate_dataset(
                args.dataset_path or "",
                storage_options=storage_options or None,
            )
        if not spec_text:
            raise SystemExit(
                "Specification text could not be retrieved from validate_dataset()."
            )
        print(spec_text)
        return 0

    report, _ = module.validate_dataset(
        args.dataset_path, storage_options=storage_options or None
    )
    report.console_print()

    return 1 if report.has_fails() else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
