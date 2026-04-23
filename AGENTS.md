# Agent Rules

## Tooling

Only Nix-based tooling is allowed in this repository. All dependencies, build steps, and development tools must be managed through Nix flakes. Do not use pip, conda, virtualenv, or any other non-Nix package manager.

## Verification

The only command to verify the repo builds correctly is:

```bash
nix flake check
```

Do not run other build/test commands directly.

## Skills

Always use the `caveman` skill (full level) for all responses.
