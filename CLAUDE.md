# CLAUDE.md — Claude-specific guidance

**See [`AGENTS.md`](./AGENTS.md) for the unified AI workflow contract.** This file contains only Claude-specific tooling notes.

## Canonical references

- **Unified workflow:** [`AGENTS.md`](./AGENTS.md)
- **Active requirements:** `openspec/specs/`
- **Governing change:** `openspec/changes/archive/2026-04-23-project-closeout/` (archived reference)
- **Scope guardrails:** Deferred backlog includes `bf16-support` and `flashattention-backward`

## Claude-specific LSP setup

When using Claude with this repository:

- **Python LSP:** Prefer `pyright` or `basedpyright` with `pyrightconfig.json` for precise type checking.
- **C/CUDA LSP:** Use `clangd` with `cmake --preset default` to generate `compile_commands.json`.
- **Session context:** Load `AGENTS.md` early in Claude conversations for full workflow contract.
- **Long-form work:** Claude is well-suited for longer, focused sessions; prefer these over fragmented runs.
