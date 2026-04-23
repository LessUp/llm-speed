# AGENTS.md

This repository uses **OpenSpec** and is currently being normalized through the **`project-closeout`** change. Every assistant working here should optimize for **stability, simplification, and high-signal cleanup**, not feature expansion.

## Canonical files

| File or directory | Purpose |
| --- | --- |
| `openspec/specs/` | Active requirements and capability definitions |
| `openspec/changes/` | Active changes and task tracking |
| `openspec/changes/project-closeout/` | Governing closeout change until archived |
| `AGENTS.md` | Shared AI workflow contract |
| `CLAUDE.md` | Claude-specific defaults |
| `.github/copilot-instructions.md` | Copilot-specific defaults |

## Project priorities

1. Keep the shipped kernels, bindings, docs, and automation coherent.
2. Remove stale or redundant structure rather than preserving low-value scaffolding.
3. Fix real defects uncovered by verification or cleanup.
4. Keep repo presentation strong: README, Pages, and GitHub About should align.

## Out of scope for closeout

- `bf16-support`
- `flashattention-backward`

These changes are **deferred backlog**. Do not treat them as release-critical unless the user explicitly re-prioritizes them.

## Required workflow

1. Read the relevant OpenSpec specs and active change artifacts before editing.
2. If work changes behavior or scope, update/create OpenSpec artifacts first.
3. Implement only the tasks that belong to the active change.
4. Keep edits deletion-first and repository-specific; avoid generic process bloat.
5. After each meaningful cleanup slice, run validation and do a review pass before merge.

## Review and branch discipline

- Use `/review` or an equivalent code-review pass before merging substantial cleanup.
- Prefer short-lived branches scoped to one OpenSpec change.
- Avoid long-lived local/cloud branch drift; merge coherent slices promptly.
- Prefer longer focused sessions over `/fleet`-style bursty runs.

## Validation commands

```bash
ruff check cuda_llm_ops/ tests/ benchmarks/
pytest tests/ -v -m "not cuda"
pre-commit run --all-files
```

Use a prepared local environment first:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt pytest hypothesis ruff pre-commit
```

## Tooling defaults

- **C/CUDA LSP:** `clangd` with `cmake --preset default` to generate `compile_commands.json`
- **Python LSP:** `pyright` or `basedpyright` using `pyrightconfig.json`
- **MCP:** keep minimal; prefer `gh`, OpenSpec commands, and targeted subagents
- **Copilot / Claude / Codex / OpenCode:** follow the same OpenSpec-first workflow; do not create tool-specific parallel processes

## File-level expectations

- `README.md` / `README.zh-CN.md`: concise project entry points
- `docs/`: durable user docs only
- `.github/workflows/`: only meaningful maintenance checks
- `CONTRIBUTING.md`: human workflow, kept aligned with OpenSpec and CLI-first usage
