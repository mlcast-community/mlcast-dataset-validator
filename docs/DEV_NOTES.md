# Documentation workflow

Use these commands from the repo root to generate and preview the spec docs.

Generate the spec pages:

```bash
uv run python docs/scripts/generate_spec_docs.py
```

Serve the site locally:

```bash
uv run mkdocs serve
```

Build the static site (outputs to `site/`):

```bash
uv run mkdocs build --strict
```
