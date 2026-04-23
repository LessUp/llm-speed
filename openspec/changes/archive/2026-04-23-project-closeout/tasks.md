## 1. Baseline and governance

- [x] 1.1 Install the documented local dependencies needed to run the repository’s existing validation commands
- [x] 1.2 Run the existing lint and CPU-safe test commands and record environment issues versus real defects
- [x] 1.3 Reconcile active OpenSpec changes so `project-closeout` becomes the governing closeout change
- [x] 1.4 Reclassify `bf16-support` and `flashattention-backward` as deferred backlog rather than closeout-critical work

## 2. Repository structure and documentation cleanup

- [x] 2.1 Remove legacy `specs/` as an active governance surface and update all internal references to OpenSpec
- [x] 2.2 Delete redundant, stale, or low-value documentation and changelog surfaces
- [x] 2.3 Rewrite the final README and docs entry surfaces around a concise, high-value information architecture
- [x] 2.4 Redesign GitHub Pages content and configuration so the site has a distinct discovery role

## 3. Workflow and AI guidance

- [x] 3.1 Rewrite `AGENTS.md` around the final OpenSpec-first closeout workflow
- [x] 3.2 Align Claude guidance and add any missing repository-level AI instruction surface needed for CLI-first use
- [x] 3.3 Add project-specific Copilot instructions and document review/subagent usage patterns
- [x] 3.4 Document lightweight LSP guidance and explicit MCP trade-offs for this repository

## 4. Automation and engineering configuration

- [x] 4.1 Simplify GitHub Actions to retain only meaningful closeout-era checks
- [x] 4.2 Tighten hooks and engineering configuration files around relevant Python and CUDA/C++ checks
- [x] 4.3 Rationalize packaging/build/config files so documented setup matches actual maintenance needs

## 5. GitHub presentation

- [x] 5.1 Align README, Pages, and repository positioning around one clear value proposition
- [x] 5.2 Update GitHub About description, homepage URL, and curated topics via `gh`

## 6. Verification and closeout

- [x] 6.1 Fix repository defects uncovered by the baseline and restructuring work
- [x] 6.2 Re-run meaningful validation from a prepared environment
- [x] 6.3 Record final known limitations and deferred work in OpenSpec rather than leaving ambiguous active drift
