# CLAUDE.md

This repository is in **closeout mode**: prefer high-signal fixes, structural cleanup, and documentation clarity over new feature expansion.

## Canonical sources

- `openspec/specs/` — active requirements
- `openspec/changes/project-closeout/` — governing closeout change until archived
- `AGENTS.md` — shared workflow contract for all AI tools
- `.github/copilot-instructions.md` — Copilot-specific guidance

## Working rules

1. Read the relevant OpenSpec spec and change artifacts before editing.
2. Do not revive the legacy `specs/` workflow or add parallel planning documents.
3. Treat `bf16-support` and `flashattention-backward` as deferred backlog, not closeout scope.
4. Keep branches short-lived and merge after coherent slices; do not let local/cloud branches drift.
5. Run a review pass after each meaningful cleanup slice before considering it merge-ready.
6. Prefer long, focused sessions over `/fleet`-style expensive bursts.

## Validation

```bash
ruff check cuda_llm_ops/ tests/ benchmarks/
pytest tests/ -v -m "not cuda"
pre-commit run --all-files
```

## Tooling defaults

- **C/CUDA LSP:** `clangd` + `cmake --preset default` for `compile_commands.json`
- **Python LSP:** `pyright`/`basedpyright` using `pyrightconfig.json`
- **MCP:** keep minimal; prefer `gh`, OpenSpec commands, and targeted subagents
- **Review:** use `/review` or an equivalent code-review pass before merge
