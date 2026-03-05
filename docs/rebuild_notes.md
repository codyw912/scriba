# Rebuild Notes

This repository is a clean-room split focused on a CLI-first pipeline.

## Intentionally rebuilt from scratch

- Any web-service layer (routes, landing pages, URL-rewrite API semantics)
- Downloader/cache/image-serving product surfaces
- Package and CLI scaffolding for this repo

## Carried-forward scope (from local pipeline work)

- Pipeline-first architecture and phase ordering
- Profile-driven orchestration and artifact state model
- Telemetry-first evaluation workflows

## Design constraints

- Backend/provider agnostic by default
- Local-file workflows first
- Fast iteration loops on small fixtures before large-doc milestone runs
