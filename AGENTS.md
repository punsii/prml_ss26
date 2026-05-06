# Agent Rules

## Project Context

This project deals with reflection/specular highlight removal from images of polished metal surfaces. A machine grinds and polishes metal, and a downstream vision model grades the surface finish quality. Strong specular reflections on the metal interfere with the grading model's accuracy. The goal is to preprocess images to suppress these reflections before grading.

## Tooling

Only Nix-based tooling is allowed in this repository. All dependencies, build steps, and development tools must be managed through Nix flakes. Do not use pip, conda, virtualenv, or any other non-Nix package manager.

## Verification

For changes to Nix files (`flake.nix`, NixOS modules), use:

```bash
nix flake check --no-build
```

`--no-build` skips build steps that the agent sandbox cannot execute, but the
evaluation pass still catches syntax errors, malformed module options, missing
attrs, and bad references. Drop `--no-build` only when running outside the
sandbox to also exercise the formatting check and the `test-specular-diffuse`
derivation.

There is **no fast verification command for Python code under `src/`** —
`flake check` does not import the streamlit app or its modules, so Python
syntax/import/runtime errors surface only when the app is actually launched.
For non-trivial Python edits, inspect carefully and, if behaviour matters,
run:

```bash
nix run .#runStreamlit
```

Do not introduce pip, conda, or other non-Nix package managers.

## Commits

**NEVER create commits unless the user explicitly orders it with words like "commit" or "commit this".**
Completing a task does NOT imply permission to commit. When in doubt, do not commit.

## Skills

Use the `caveman` skill (light level) for the main model.
Use the `caveman` skill (full level) for all responses on Anthropic model (opus/sonnet) subagents.
Exception: The `local` (Ollama) subagent has skills disabled to preserve context budget.
