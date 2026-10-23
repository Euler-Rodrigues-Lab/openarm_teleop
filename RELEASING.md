# Releasing openarm-teleop

## One-time PyPI configuration

Create a PyPI account and enable two-factor authentication. In your PyPI account,
add a pending publisher with:

- PyPI project: `openarm-teleop`
- Owner: `Euler-Rodrigues-Lab`
- Repository: `openarm_teleop`
- Workflow: `release.yml`
- Environment: `pypi`

Create the matching `pypi` GitHub environment and configure a required reviewer.
The workflow uses OIDC Trusted Publishing; do not store a PyPI password in GitHub.

## Release procedure

1. Run the repository tests. Bump `project.version` for every changed release
   (also update `xrt_devices.__version__` for XRT_devices).
2. Build with `uv build --no-sources`, check with `uvx --from twine twine check dist/*`,
   and test the installed wheel from outside the checkout.
3. Commit the reviewed changes and push the matching `v<version>` tag.
4. Approve the `pypi` environment deployment. PyPI artifacts are immutable:
   fix mistakes with a new version, never replace a published wheel.

Publish geo-kin-core and xrt-devices before openarm-teleop. OpenArm 0.1.1 pins
both dependencies to the tested 0.1.1 release. After those dependencies are live,
run `uv lock` in OpenArm and commit the generated lock before tagging OpenArm.

Only public Python packages belong on PyPI. Never include private `geo_kin`
solver wheels, signed organization licenses, or signing keys in these releases.
