# Contributing

This repo uses a simple branch + PR workflow.

## Branch Strategy

- `main` is the integration branch and should stay releasable.
- Do not do feature work directly on `main`.
- Use short-lived topic branches:
  - `feature/<topic>`
  - `fix/<topic>`
  - `chore/<topic>`
  - `docs/<topic>`

## Daily Flow

```bash
git fetch origin
git switch main
git pull --ff-only
git switch -c feature/<topic>
```

Make changes, run checks, then commit and push:

```bash
uv run --with pytest --with pytest-asyncio python -m pytest -q
git add -A
git commit -m "<type>: <short summary>"
git push -u origin feature/<topic>
```

Open a PR to `main` and squash-merge after review.

## Commit Style

Use concise commit messages with a type prefix:

- `feature:` new functionality
- `fix:` bug fix
- `chore:` maintenance/refactor/tooling
- `docs:` documentation only
- `test:` tests only

Example: `feature: add deterministic benchmark PDF variant generator`

## PR Expectations

Every PR should include:

- purpose and scope,
- key implementation notes,
- validation performed (commands + outcome),
- follow-up items if the work is partial.

Keep PRs focused and reasonably small when possible.

## Checks Before Merge

- tests pass locally,
- docs updated for behavior/config changes,
- no secrets committed (`.env`, API keys, local caches),
- no large generated artifacts unless explicitly intended.

## Confidence Levels

- Unit/helper coverage: `uv run -m pytest`
- Deterministic integration coverage: end-to-end benchmark/report flow under
  `tests/test_integration_benchmark_flow.py` plus the installed-wheel smoke in CI
- Live provider confidence: run the `Live Integration` GitHub Actions workflow
  with `OPENROUTER_API_KEY` before trusting benchmark/report changes that depend
  on real model behavior
- The `Live Integration` workflow is manual-only and skips cleanly when the
  secret is not configured; it should not gate normal pushes.

## Live API / Benchmark Runs

- Prefer preserving artifacts for paid/live runs even when validation reports hard errors.
- Use strict fail mode (`fail_on_hard_errors: true`) primarily for CI/profile hard gates.
- Capture run metadata and outcomes in the matrix/report artifacts.
- Treat live provider smoke runs as the final confidence check, not a replacement
  for deterministic tests.
