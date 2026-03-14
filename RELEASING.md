# Releasing

Use this checklist before publishing a new `scribai` release to PyPI.

## Release bar for `v0.1.x`

- Core CLI flow works from an installed wheel:
  - `scribai run --input ...`
  - `scribai doctor --input ...`
  - `scribai status --run-id ...`
- Runtime-native home behavior works without the source tree.
- `--output` copies user-facing final exports successfully.
- Built-in presets work with a configured provider API key.
- Passthrough mode works without any provider credentials.

## Pre-release checks

Run from the repository root:

```bash
uv run -m pytest
uv build
uv tool run --from dist/*.whl scribai --help
uv tool run --from dist/*.whl scribai doctor --preset passthrough --input samples/docs/mini_api.md
uv tool run --from dist/*.whl scribai run --preset passthrough --input samples/docs/mini_api.md --run-id release-smoke --output /tmp/scribai-release-smoke
```

Optional provider-backed smoke test:

```bash
OPENROUTER_API_KEY=... uv tool run --from dist/*.whl scribai run --input samples/docs/mini_api.md --run-id release-provider-smoke
```

Or run the `Live Integration` GitHub Actions workflow for a provider-backed CLI
and matrix/report smoke check before tagging a release.

## Release automation

- CI runs on pushes/PRs and covers:
  - `uv run -m pytest`
  - `uv build`
  - installed-wheel smoke checks for `scribai --help`, `doctor`, `run`, and
    `status`
- Publish automation runs on pushed version tags matching `v*`.
- PyPI publishing is intended to use GitHub trusted publishing via
  `pypa/gh-action-pypi-publish`.

Before the first publish, configure the PyPI project to trust this GitHub
repository/workflow.

## Publish flow

Build artifacts locally if you want a pre-publish sanity check:

```bash
uv build
```

Check package metadata/rendering:

```bash
uvx twine check dist/*
```

Optional manual upload to TestPyPI first:

```bash
uvx twine upload --repository testpypi dist/*
```

If release automation is configured, pushing `v0.1.0` will publish to PyPI.

Optional manual upload to PyPI:

```bash
uvx twine upload dist/*
```

## Git/tag steps

Create and push an annotated tag after merge to `main`:

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```

## Notes

- Keep benchmark/report experimentation out of the release bar unless it is part
  of the public `scribai` CLI contract.
- Prefer small patch releases after PyPI publish instead of batching unrelated
  changes into the initial `v0.1.0` cut.
