# Agent Rules

## Project Context

This project deals with reflection/specular highlight removal from images of polished metal surfaces. A machine grinds and polishes metal, and a downstream vision model grades the surface finish quality. Strong specular reflections on the metal interfere with the grading model's accuracy. The goal is to preprocess images to suppress these reflections before grading.

## Tooling

Only Nix-based tooling is allowed in this repository. All dependencies, build steps, and development tools must be managed through Nix flakes. Do not use pip, conda, virtualenv, or any other non-Nix package manager.

## Verification

The only command to verify the repo builds correctly is:

```bash
nix flake check
```

Do not run other build/test commands directly.

## Commits

**NEVER create commits unless the user explicitly orders it with words like "commit" or "commit this".**
Completing a task does NOT imply permission to commit. When in doubt, do not commit.

## Skills

Use the `caveman` skill (light level) for the main model.
Use the `caveman` skill (full level) for all responses on Anthropic model (opus/sonnet) subagents.
Exception: The `local` (Ollama) subagent has skills disabled to preserve context budget.
