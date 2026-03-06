## Profile Layout

- `profiles/pipeline.profile.example.yaml`: minimal baseline profile
- `profiles/local_attached/`: profiles that connect to an already-running local backend
- `profiles/local_spawned/`: profiles where scriba launches and manages the backend process
- `profiles/remote/`: hosted provider profiles (for example OpenRouter)
- `profiles/hybrid/`: mixed-topology profiles

All scripts and docs now reference profile paths from this directory structure.

## Adapter standard

- Canonical adapter is `litellm` for remote/attached/spawned model calls.

## OCR role contract

- `ocr_vision` is an optional role used by the `extract` stage for PDF OCR.
- If `ocr_vision` is configured, `extract` routes PDF pages through that model.
- If `ocr_vision` is missing (or OCR call fails), extraction falls back to local
  `pymupdf4llm` text extraction.

Current remote/hybrid examples use GLM-OCR (`provider: glm_ocr`, `model: glm-ocr`)
as the default OCR anchor.

To change OCR backend/model, edit:

- `backends.ocr_backend`
- `roles.ocr_vision.model`
