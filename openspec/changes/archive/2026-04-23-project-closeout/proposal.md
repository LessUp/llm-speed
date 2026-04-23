## Why

The repository is mid-migration to OpenSpec but still contains multiple competing sources of truth across legacy specs, documentation, GitHub Pages, AI instruction files, and workflow configuration. This drift is slowing closeout, hiding real defects behind noise, and making the project look less credible than the implementation warrants.

The project goal is now stabilization and finish-quality, not capability expansion. This change creates the governance, cleanup, and presentation requirements needed to aggressively normalize the repository for archive-ready maintenance.

## What Changes

- Make `openspec/` the only active specification and workflow authority for future changes.
- Remove legacy `specs/`-driven governance and eliminate redundant or stale documentation surfaces.
- Rebuild the documentation set so README, Pages, changelog, and contributor guidance each have a distinct purpose.
- Simplify engineering automation, hooks, and project configuration so only meaningful checks remain.
- Add CLI-first workflow guidance for Claude, Codex, Copilot, review checkpoints, and subagent usage.
- Standardize lightweight tooling strategy for LSP and MCP so the repository favors high-signal defaults over context-heavy integrations.
- Reposition GitHub Pages and GitHub About metadata to present the project clearly to new users.
- Explicitly downgrade unfinished feature proposals that are not part of the closeout target.

## Capabilities

### New Capabilities
- `repository-governance`: Defines the final OpenSpec-only governance model, repo structure, and closeout backlog handling.
- `documentation-surface`: Defines the minimal high-value documentation set, Pages role, and changelog discipline.
- `engineering-workflow`: Defines automation, hooks, AI instructions, CLI-first development workflow, and tooling guidance.
- `project-presentation`: Defines the repository positioning across README, GitHub Pages, and GitHub About metadata.

### Modified Capabilities
- `python-interface`: Clarify that closeout verification MUST distinguish environment/setup failures from product defects before treating verification as failed.

## Impact

- Affected areas: `openspec/`, `README*`, `docs/`, GitHub Pages files, `.github/workflows/`, hooks, packaging/config files, AI instruction files, and GitHub repository metadata.
- Affected systems: contributor workflow, AI-assisted workflow, documentation publishing, and CI signal quality.
- No new product runtime features are introduced; the focus is repository governance, usability, and closeout reliability.

## Success Criteria

- OpenSpec becomes the only active governance layer and legacy `specs/` no longer acts as a source of truth.
- Repository docs are reduced to a small, coherent, non-duplicative set with a distinct Pages landing experience.
- GitHub Actions, hooks, and config files are simplified to meaningful closeout-era checks.
- GitHub About description, homepage, and topics align with the final project positioning.
- Existing validation commands run from a prepared environment and any uncovered defects are fixed or explicitly backlogged in OpenSpec.
