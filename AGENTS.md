# AGENTS.md — Unified AI Workflow Contract

This repository uses **OpenSpec** and is currently being normalized through the **`project-closeout`** change. Every assistant working here—including Claude, Copilot, Codex, and other LLM-based tools—should optimize for **stability, simplification, and high-signal cleanup**, not feature expansion.

## Canonical sources

| File or directory | Purpose |
| --- | --- |
| `openspec/specs/` | Active requirements and capability definitions |
| `openspec/changes/archive/2026-04-23-project-closeout/` | Governing closeout change (archived reference) |
| `AGENTS.md` | **Unified AI workflow contract (this file)** |
| `CLAUDE.md` | Claude-specific tooling notes (thin delta to AGENTS.md) |
| `.github/copilot-instructions.md` | Copilot-specific tooling notes (thin delta to AGENTS.md) |
| `.claude/CLAUDE.md` | Mirror pointer to `CLAUDE.md` |

## Project scope and priorities

### Out of scope for closeout

- `bf16-support`
- `flashattention-backward`

These changes are **deferred backlog**. Do not treat them as release-critical unless the user explicitly re-prioritizes them.

### In scope

1. Keep the shipped kernels, bindings, docs, and automation coherent.
2. Remove stale or redundant structure rather than preserving low-value scaffolding.
3. Fix real defects uncovered by verification or cleanup.
4. Keep repo presentation strong: README, Pages, and GitHub About should align.

## Shared workflow rules for all tools

### Before editing

1. Read the relevant OpenSpec specs and active change artifacts.
2. If work changes behavior or scope, update/create OpenSpec artifacts first.
3. Implement only the tasks that belong to the active change.

### After editing

4. Keep edits deletion-first and repository-specific; avoid generic process bloat.
5. After each meaningful cleanup slice, run validation and do a review pass before merge.

### Branch and merge discipline

- Use `/review` or an equivalent code-review pass before merging substantial cleanup.
- Prefer short-lived branches scoped to one OpenSpec change.
- Avoid long-lived local/cloud branch drift; merge coherent slices promptly.
- Prefer longer focused sessions over `/fleet`-style bursty runs.

## Validation and setup

### Validation commands

All tools should run these checks before claiming changes are ready:

```bash
ruff check cuda_llm_ops/ tests/ benchmarks/
pytest tests/ -v -m "not cuda"
pre-commit run --all-files
```

### Local environment setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt pytest hypothesis ruff pre-commit
```

## Shared tooling defaults

### LSP and development tools

- **C/CUDA LSP:** `clangd` with `cmake --preset default` to generate `compile_commands.json`
- **Python LSP:** `pyright` or `basedpyright` using `pyrightconfig.json`
- **Git and task tracking:** `gh` CLI, OpenSpec commands, targeted subagents (prefer over heavyweight MCP)
- **Pre-commit hooks:** Run `pre-commit run --all-files` before merge

### MCP principle

Keep MCP minimal and repository-specific. Prefer `gh`, OpenSpec commands, and targeted subagents over generic plugin infrastructure.

## File-level expectations

- `README.md` / `README.zh-CN.md`: concise project entry points
- `docs/`: durable user docs only
- `.github/workflows/`: only meaningful maintenance checks
- `CONTRIBUTING.md`: human workflow, kept aligned with OpenSpec and CLI-first usage

## Tool-specific guidance

**All tools (Claude, Copilot, Codex, etc.):** Follow the unified contract above. Do not create tool-specific parallel planning documents or processes.

See `CLAUDE.md` and `.github/copilot-instructions.md` for thin delta guidance on LSP/tooling preferences.
