# Copilot instructions for LLM-Speed

**See [`AGENTS.md`](../../AGENTS.md) for the unified AI workflow contract.** This file contains only Copilot-specific guidance.

## Unified workflow contract

The repository uses **OpenSpec** and is in **closeout mode**. Copilot should follow the complete workflow defined in `AGENTS.md`, including:

- **Optimization:** Stability, simplification, and high-signal cleanup over features
- **Canonical sources:** `openspec/specs/`, `openspec/changes/archive/2026-04-23-project-closeout/`
- **Scope:** Deferred backlog includes `bf16-support` and `flashattention-backward`
- **Validation:** `ruff check`, `pytest -m "not cuda"`, `pre-commit run --all-files`

## Copilot-specific execution guidance

### Workflow

1. Read relevant OpenSpec specs and active change tasks before starting.
2. Implement the smallest coherent slice per change.
3. Update docs/config/tests that are directly affected.
4. Run validation commands before claiming readiness.
5. Perform a `/review` or equivalent code-review pass before merge.

### Tooling hints

- Generate `compile_commands.json` with `cmake --preset default` for `clangd`.
- Use the local `.venv` plus `pyrightconfig.json` for Python language tooling.
- Keep MCP minimal; prefer `gh`, OpenSpec commands, and targeted subagents.

## Guardrails

- Do not reintroduce the legacy `specs/` hierarchy.
- Avoid heavyweight MCP additions unless they provide clear repository-specific value.
- Do not add speculative workflows, plugins, or config files that the repository will not maintain.
