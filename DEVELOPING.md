# Editable development

Customer installations use published wheels. To work on the shared libraries,
clone them alongside this project and explicitly install them as editable:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ../geo_kin_core -e ../XRT_devices -e '.[test]'
.venv/bin/python -m pytest -q
```

Use dependency checkouts at version 0.1.1 to satisfy this release's pins.
Run `.venv/bin/python` directly during editable development; `uv sync` restores
the project's published dependencies. The historical submodules remain available
for reference but are not the default install source and may be older versions.

After publishing both dependency wheels to PyPI, run `uv lock` and commit the
new lockfile. The old lock referenced editable submodules and is removed in this
migration. The release order is documented in RELEASING.md.
