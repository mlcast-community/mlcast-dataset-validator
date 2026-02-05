#!/usr/bin/env python3
"""Generate MkDocs pages for each MLCast dataset specification."""
from __future__ import annotations

import importlib
import pkgutil
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List

from mlcast_dataset_validator.specs.reporting import skip_all_checks

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPEC_PACKAGE = "mlcast_dataset_validator.specs"
DOCS_DIR = REPO_ROOT / "docs"
SPECS_DIR = DOCS_DIR / "specs"


def _run_git(args: List[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _get_repo_url() -> str | None:
    url = _run_git(["remote", "get-url", "origin"]) or _run_git(
        ["remote", "get-url", "upstream"]
    )
    if not url:
        return None
    if url.endswith(".git"):
        url = url[: -len(".git")]
    ssh_match = re.match(r"git@github.com:(.+/.+)", url)
    if ssh_match:
        url = f"https://github.com/{ssh_match.group(1)}"
    return url


def _get_default_branch() -> str:
    head = _run_git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"]) or ""
    if head.startswith("origin/"):
        return head.split("/", 1)[1]
    return "main"


def _discover_catalog() -> Dict[str, List[str]]:
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
    title = f"{data_stage}/{product} specification"
    repo_link = ""
    if repo_url:
        path = f"mlcast_dataset_validator/specs/{data_stage}/{product}.py"
        spec_url = f"{repo_url}/blob/{default_branch}/{path}"
        repo_link = f"[View spec source on GitHub]({spec_url})"

    command_run = (
        "uvx --with mlcast-dataset-validator mlcast.validate_dataset "
        f"{data_stage} {product} <PATH_OR_URL>"
    )

    frontmatter, body = _extract_frontmatter(spec_text)
    spec_version = frontmatter.get("version", "unknown")

    parts = [
        f"# {title}",
        "",
    ]
    if repo_link:
        parts.extend([repo_link, ""])

    parts.extend(
        [
            "Data stage:",
            f"`{data_stage}`",
            "",
            "Product:",
            f"`{product}`",
            "",
            "Spec version:",
            f"`{spec_version}`",
            "",
            "Run the validator:",
            "",
            "```bash",
            command_run,
            "```",
            "",
            "---",
            "",
            body.strip(),
            "",
        ]
    )
    return "\n".join(parts)


def _write_index(catalog: Dict[str, List[str]]) -> None:
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
        for product in catalog[data_stage]:
            link = f"specs/{data_stage}/{product}.md"
            lines.append(f"- [{data_stage}/{product}]({link})")
    lines.append("")
    (DOCS_DIR / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    catalog = _discover_catalog()
    repo_url = _get_repo_url()
    default_branch = _get_default_branch()

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
