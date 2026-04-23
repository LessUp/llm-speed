## ADDED Requirements

### Requirement: Automation remains high-signal
The repository SHALL keep only automation that provides meaningful maintenance or release confidence for the closeout-era project.

#### Scenario: Workflow or hook is evaluated
- **WHEN** a CI workflow, pre-commit hook, or engineering check is retained in the repository
- **THEN** it MUST correspond to a concrete quality gate that contributors are expected to run or rely on

### Requirement: CLI-first development workflow is documented
The repository SHALL document a CLI-first workflow for OpenSpec-driven work across Claude, Codex, and Copilot usage.

#### Scenario: AI-assisted contributor starts work
- **WHEN** a contributor uses AI tooling to plan, implement, or review a change
- **THEN** the project guidance MUST describe how to use OpenSpec, review checkpoints, and subagent/delegation patterns in a consistent workflow

### Requirement: Tooling guidance stays lightweight
The repository SHALL recommend lightweight, reusable tooling defaults and MUST avoid requiring heavyweight local integrations unless they provide repository-specific value.

#### Scenario: Contributor chooses LSP or MCP setup
- **WHEN** tooling guidance discusses LSP servers, MCP integrations, or plugins
- **THEN** the guidance MUST prefer low-overhead defaults, explain tool-agnostic reuse where applicable, and state when heavier integrations are unnecessary

### Requirement: Hooks are fast and targeted
The repository SHALL use local hooks that are fast enough for routine use and tightly scoped to the repository’s actual languages and checks.

#### Scenario: Contributor installs hooks
- **WHEN** a contributor enables repository hooks
- **THEN** the hooks MUST focus on relevant Python and C++/CUDA quality checks plus basic file hygiene rather than broad generic automation

### Requirement: Review checkpoints are part of the workflow
The repository SHALL define explicit points where contributors use review-oriented analysis before merging major cleanup work.

#### Scenario: Contributor completes a substantial closeout change
- **WHEN** the contributor finishes a logically complete cleanup or restructuring slice
- **THEN** the documented workflow MUST require a review step before final merge or archival actions
