# Contributing

This project uses **OpenSpec**. All meaningful changes start from specs and tracked changes under `openspec/`.

## Start here

1. Read the relevant capability spec in `openspec/specs/`.
2. Check active changes in `openspec/changes/`.
3. If your work changes behavior or scope, create or update a change proposal before editing code.
4. Keep the branch focused and short-lived.
5. Run validation and a review pass before merge.

For AI-specific guidance, see [`AGENTS.md`](AGENTS.md).

## Local setup

```bash
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

python3 -m venv .venv
. .venv/bin/activate

pip install -U pip setuptools wheel
pip install -r requirements.txt pytest hypothesis ruff pre-commit
```

To build the CUDA extension locally:

```bash
pip install -e .
```

## Validation

```bash
ruff check cuda_llm_ops/ tests/ benchmarks/
pytest tests/ -v -m "not cuda"
pre-commit run --all-files
```

GPU-specific validation can be run separately with:

```bash
pytest tests/ -v -m cuda
```

## Workflow

### OpenSpec commands

```bash
/opsx:propose <change-name>
/opsx:apply <change-name>
/opsx:archive <change-name>
```

### Branch and review discipline

- Keep one branch per change or cleanup slice.
- Merge promptly after a coherent slice; avoid stale local/cloud branch drift.
- Use `/review` or an equivalent review pass before merge.
- Prefer concise, project-specific docs and config over generic process sprawl.

### Commit style

Use conventional commits such as:

```text
fix(ci): simplify cpu-safe validation
docs(readme): align setup with OpenSpec workflow
```

## Tooling

- **C/CUDA:** `clangd` + `cmake --preset default`
- **Python:** `pyright`/`basedpyright` + `pyrightconfig.json`
- **Hooks:** `pre-commit` with Ruff, clang-format, and file hygiene
- **MCP:** optional; prefer `gh`, OpenSpec commands, and targeted subagents unless a heavier integration clearly pays off

## Current closeout scope

The repository is being normalized for final stabilization. `bf16-support` and `flashattention-backward` remain deferred backlog and are not part of the active closeout path.
