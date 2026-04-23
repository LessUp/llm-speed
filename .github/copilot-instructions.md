# Copilot instructions for LLM-Speed

This repository is managed with **OpenSpec** and is currently being normalized through the `project-closeout` change.

## What Copilot should optimize for

- Favor **stability, simplification, and clarity** over new features.
- Use `openspec/specs/` and `openspec/changes/` as the only active planning sources.
- Keep documentation concise and project-specific; remove low-value duplication.
- Preserve a **CLI-first** workflow for Claude, Codex, Copilot CLI, and similar tools.
- Recommend `/review` after each coherent cleanup slice.

## Scope guardrails

- `bf16-support` and `flashattention-backward` are **deferred backlog**, not closeout deliverables.
- Do not reintroduce the legacy `specs/` hierarchy.
- Avoid heavyweight MCP additions unless they provide clear repository-specific value.
- Do not add speculative workflows, plugins, or config files that the repository will not maintain.

## Repository-specific workflow

1. Read the relevant OpenSpec specs and active change tasks.
2. Implement the smallest coherent slice.
3. Update docs/config/tests that are directly affected.
4. Run existing validation commands.
5. Perform a review pass before merge.

## Preferred validation commands

```bash
ruff check cuda_llm_ops/ tests/ benchmarks/
pytest tests/ -v -m "not cuda"
pre-commit run --all-files
```

## Tooling hints

- Generate `compile_commands.json` with `cmake --preset default` for `clangd`.
- Use the local `.venv` plus `pyrightconfig.json` for Python language tooling.
