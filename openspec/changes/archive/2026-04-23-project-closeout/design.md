## Context

The repository already contains partially completed migration work: `openspec/` exists, three active changes exist, and several top-level docs were edited but not fully reconciled. At the same time, legacy `specs/` content still appears in docs navigation, contributor guidance, and Pages configuration. The result is duplicated governance, duplicated documentation, and a project presentation layer that overstates structure without increasing clarity.

This change is intentionally cross-cutting. It touches OpenSpec, user-facing docs, publishing, contributor workflow, engineering automation, and GitHub metadata. Because the project is being prepared for a low-maintenance closeout phase, the design must optimize for clarity, low future upkeep, and strong defaults for AI-assisted CLI workflows.

## Goals / Non-Goals

**Goals:**
- Establish a single governance model centered on OpenSpec.
- Reduce the repository to a deliberate, maintainable document and config set.
- Preserve user-visible value while deleting low-signal, duplicated, or stale material.
- Make automation fast, comprehensible, and worth keeping.
- Encode a practical CLI-first workflow for Claude, Codex, and Copilot users.
- Leave unfinished feature ambitions clearly deferred instead of ambiguously active.

**Non-Goals:**
- Implement BF16 support or FlashAttention backward support in this change.
- Expand runtime APIs beyond what is required to fix closeout-blocking defects.
- Add heavy infrastructure solely for process aesthetics.
- Optimize for large multi-IDE teams; the default workflow is CLI-first and minimal.

## Decisions

### 1. OpenSpec becomes the only active governance layer

**Decision:** Treat `openspec/specs/` and `openspec/changes/` as the only active planning and requirements surface. Legacy `specs/` content will be removed or reduced to a minimal redirect surface if needed.

**Why:** The current split between `openspec/` and legacy `specs/` causes conflicting workflow advice and duplicate navigation in Pages and docs.

**Alternative considered:** Keep both structures and document when to use each. Rejected because it preserves ambiguity and increases maintenance cost.

### 2. Use deletion-first cleanup instead of archival sprawl

**Decision:** Remove redundant docs, duplicated changelog surfaces, and obsolete config when they no longer serve the final repo shape.

**Why:** The repo goal is closeout. Keeping large amounts of low-value historical scaffolding makes the project harder to trust and maintain.

**Alternative considered:** Archive every displaced file under more history folders. Rejected because the repository already has enough historical trace via git.

### 3. Separate documentation surfaces by intent

**Decision:** Keep a strong root README for repo conversion, a concise but distinctive Pages landing surface for discovery, focused contributor/AI workflow docs, and one meaningful changelog strategy.

**Why:** Current surfaces repeat each other. Pages should market and orient; README should convert and instruct; workflow docs should govern contribution.

**Alternative considered:** Make Pages a direct mirror of README. Rejected because it offers no additional value to visitors.

### 4. Simplify automation to high-signal checks only

**Decision:** Keep lightweight lint/test/documentation validation that directly protects the stabilized repo. Remove gratuitous complexity and brittle pseudo-checks.

**Why:** Low-value workflows create maintenance noise and wasted CI runs, especially for an archive-leaning project.

**Alternative considered:** Add more exhaustive workflows for every surface. Rejected because the project is in closeout mode, not expansion mode.

### 5. Favor CLI-first AI guidance and lightweight tooling

**Decision:** Document one coherent workflow for OpenSpec + Claude/Codex/Copilot CLI, using `/review` and subagents selectively. Recommend lightweight LSP support and explicitly avoid mandatory heavy MCP stacks unless they offer repository-specific leverage.

**Why:** The primary operating mode is not IDE-centric, and context-heavy tooling would increase cost without proportionate value.

**Alternative considered:** Add broad IDE-specific config and generic MCP recommendations. Rejected because it mismatches actual usage.

### 6. Convert unfinished feature proposals into explicit deferred work

**Decision:** Keep `bf16-support` and `flashattention-backward` out of the closeout critical path and mark them as deferred backlog rather than ambiguous in-progress work.

**Why:** Leaving them active implies product expansion is still a release priority, which conflicts with the closeout objective.

**Alternative considered:** Finish both before cleanup. Rejected due to scope expansion and timeline risk.

## Risks / Trade-offs

- **[Risk]** Large-scale deletion could remove context users still expect. → **Mitigation:** keep the final remaining surfaces highly informative and ensure GitHub pages/README cover essential entry points.
- **[Risk]** Refactoring workflow docs may conflict with uncommitted local migration edits. → **Mitigation:** work with existing edits in place and avoid reverting unrelated user work.
- **[Risk]** Simplifying CI may drop checks that still catch useful failures. → **Mitigation:** preserve only commands that correspond to real maintenance needs and existing project validation.
- **[Risk]** Closeout cleanup could uncover real runtime or packaging defects. → **Mitigation:** establish a prepared environment baseline early and fix concrete failures before concluding the change.

## Migration Plan

1. Establish a working local baseline by installing documented dependencies and running existing checks.
2. Reconcile OpenSpec artifacts and introduce closeout governance requirements.
3. Delete legacy governance/docs surfaces and rebuild the remaining ones around the new requirements.
4. Simplify workflows, hooks, and tooling configs to match the closeout model.
5. Rebuild README/Pages/About metadata so external presentation matches the final project narrative.
6. Re-run meaningful validation and explicitly defer non-closeout feature work in OpenSpec.

## Open Questions

- Whether a tiny compatibility README should remain under legacy `specs/` for external links, or whether full removal is preferable once all internal references are updated.
- Whether repo-local LSP settings should be expressed via `.clangd`/lightweight config only, or documented without committed editor config files if the repo stays fully CLI-first.
