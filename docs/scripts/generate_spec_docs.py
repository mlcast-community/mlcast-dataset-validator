#!/usr/bin/env python3
"""Generate MkDocs pages for each MLCast dataset specification."""
from __future__ import annotations

import importlib
import pkgutil
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List

from packaging.version import Version

from mlcast_dataset_validator import __version__
from mlcast_dataset_validator.specs.reporting import skip_all_checks

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPEC_PACKAGE = "mlcast_dataset_validator.specs"
DOCS_DIR = REPO_ROOT / "docs"
SPECS_DIR = DOCS_DIR / "specs"


REPO_URL = "https://github.com/mlcast-community/mlcast-dataset-validator"
DEFAULT_BRANCH = "main"


def _discover_catalog() -> Dict[str, List[str]]:
    """Return available data_stage/product combinations under specs/.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, list[str]]
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


def _extract_frontmatter(spec_text: str) -> tuple[dict, str]:
    """Parse YAML front matter and return metadata plus remaining Markdown.

    Parameters
    ----------
    spec_text : str
        Raw spec Markdown text that may start with YAML front matter.

    Returns
    -------
    tuple[dict, str]
        Parsed key/value metadata and the remaining Markdown body.
    """
    text = textwrap.dedent(spec_text).lstrip()
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not match:
        return {}, text
    frontmatter = {}
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        frontmatter[key.strip()] = value.strip()
    remainder = text[match.end() :].lstrip()
    return frontmatter, remainder


def _render_spec_page(
    data_stage: str,
    product: str,
    spec_text: str,
    repo_url: str | None,
    default_branch: str,
) -> str:
    """Render a full Markdown page for a single spec.

    Parameters
    ----------
    data_stage : str
        Data stage name (e.g., ``source_data``).
    product : str
        Product identifier (e.g., ``radar_precipitation``).
    spec_text : str
        Raw spec Markdown text.
    repo_url : str | None
        Base repository URL used to link to spec sources.
    default_branch : str
        Default branch name used in source links.

    Returns
    -------
    str
        Rendered Markdown page content.
    """
    title = (
        f"{product.replace('_', ' ').capitalize()} " f"{data_stage.replace('_', ' ')}"
    )
    repo_link = ""
    if repo_url:
        path = f"mlcast_dataset_validator/specs/{data_stage}/{product}.py"
        spec_url = f"{repo_url}/blob/{default_branch}/{path}"
        repo_link = f"[View spec source on GitHub]({spec_url})"

    command_run = (
        f"uvx --with mlcast-dataset-validator@{__version__} mlcast.validate_dataset "
        f"{data_stage} {product} <PATH_OR_URL>"
    )
    command_source = (
        'uvx --from "git+https://github.com/mlcast-community/mlcast-sourcedata-validator" '
        f"mlcast.validate_dataset {data_stage} {product} <PATH_OR_URL>"
    )
    version = Version(__version__)
    show_dirty_warning = version.is_prerelease or version.local is not None

    frontmatter, body = _extract_frontmatter(spec_text)

    parts = [
        f"# {title}",
        "",
    ]

    frontmatter_block = None
    if frontmatter:
        yaml_lines = []
        for key in sorted(frontmatter):
            yaml_lines.append(f"{key}: {frontmatter[key]}")
        frontmatter_block = "\n".join(yaml_lines)

    if frontmatter_block:
        parts.extend(
            [
                "```yaml",
                frontmatter_block,
                "```",
                "",
            ]
        )

    if repo_link:
        parts.extend([repo_link, ""])

    parts.extend(
        [
            "Run the validator directly with "
            "[uv](https://docs.astral.sh/uv/getting-started/installation/) "
            "from [release on pypi.org](https://pypi.org/project/mlcast-dataset-validator/):",
            "",
            "```bash",
            command_run,
            "```",
            "",
        ]
    )
    if show_dirty_warning:
        parts.extend(
            [
                (
                    "> Warning: this build uses a pre-release or local version "
                    f"(`{__version__}`), which means `main` may include changes not yet "
                    "released on PyPI. You can run directly from the GitHub source instead:"
                ),
                ">",
                f">     {command_source}",
                "",
            ]
        )
    parts.extend(
        [
            "---",
            "",
            body.strip(),
            "",
        ]
    )
    return "\n".join(parts)


def _write_index(catalog: Dict[str, List[str]]) -> None:
    """Write docs/index.md with links to each generated spec page.

    Parameters
    ----------
    catalog : dict[str, list[str]]
        Mapping of data_stage names to product module names.

    Returns
    -------
    None
    """
    lines = [
        "# MLCast dataset specifications",
        "",
        "This site hosts the rendered specifications for the MLCast dataset validator.",
        "Each spec describes the required structure and metadata for a given data stage",
        "and product, and provides the exact validation requirements used by the CLI.",
        "",
        "Available specifications:",
        "",
    ]
    for data_stage in sorted(catalog):
        stage_label = data_stage.replace("_", " ").capitalize()
        lines.append(f"- {stage_label}")
        for product in catalog[data_stage]:
            product_label = product.replace("_", " ").capitalize()
            link = f"specs/{data_stage}/{product}.md"
            lines.append(f"    - [{product_label}]({link})")
    lines.append("")
    (DOCS_DIR / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _write_mkdocs_nav(catalog: Dict[str, List[str]]) -> None:
    """Update mkdocs.yml to include generated spec pages in the nav.

    Parameters
    ----------
    catalog : dict[str, list[str]]
        Mapping of data_stage names to product module names.

    Returns
    -------
    None
    """
    mkdocs_path = REPO_ROOT / "mkdocs.yml"
    text = mkdocs_path.read_text(encoding="utf-8")
    begin = "# -- specs-nav-begin --"
    end = "# -- specs-nav-end --"
    if begin not in text or end not in text:
        raise SystemExit("mkdocs.yml is missing specs nav markers.")

    nav_lines = []
    for data_stage in sorted(catalog):
        stage_label = data_stage.replace("_", " ").capitalize()
        nav_lines.append(f"  - {stage_label}:")
        for product in catalog[data_stage]:
            product_label = product.replace("_", " ").capitalize()
            path = f"specs/{data_stage}/{product}.md"
            nav_lines.append(f"      - {product_label}: {path}")
    nav_block = "\n".join(nav_lines) if nav_lines else "  - Specs: index.md"

    before, remainder = text.split(begin, 1)
    _, after = remainder.split(end, 1)
    updated = "".join([before, begin, "\n", nav_block, "\n", end, after])
    mkdocs_path.write_text(updated, encoding="utf-8")


def main() -> int:
    """Generate spec pages and the index page for MkDocs.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Exit status code, zero on success.
    """
    catalog = _discover_catalog()
    repo_url = REPO_URL
    default_branch = DEFAULT_BRANCH

    SPECS_DIR.mkdir(parents=True, exist_ok=True)

    for data_stage, products in catalog.items():
        for product in products:
            module_name = f"{SPEC_PACKAGE}.{data_stage}.{product}"
            module = importlib.import_module(module_name)
            with skip_all_checks():
                _, spec_text = module.validate_dataset("")
            if not spec_text:
                raise SystemExit(
                    f"Spec text missing for {data_stage}/{product}; cannot render docs."
                )

            output_dir = SPECS_DIR / data_stage
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{product}.md"
            output_path.write_text(
                _render_spec_page(
                    data_stage=data_stage,
                    product=product,
                    spec_text=spec_text,
                    repo_url=repo_url,
                    default_branch=default_branch,
                ),
                encoding="utf-8",
            )

    _write_index(catalog)
    _write_mkdocs_nav(catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
