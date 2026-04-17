# RFC (Request for Comments) Documents

This directory contains technical design documents, architecture decisions, and implementation plans.

---

## Purpose

RFC documents define:

- **How** features should be implemented
- **Why** specific technical decisions were made
- **What** constraints and trade-offs exist

---

## Current RFCs

| RFC | Status | Title | Description |
|-----|--------|-------|-------------|
| [0001](0001-core-architecture.md) | ✅ Accepted | Core Architecture | Technical design and kernel specifications |
| [0002](0002-implementation-tasks.md) | ✅ Accepted | Implementation Tasks | Task breakdown and progress tracking |

---

## RFC Naming Convention

```
NNNN-short-title.md

NNNN: Sequential number (0001, 0002, ...)
short-title: Kebab-case title
```

---

## RFC Lifecycle

| Status | Description |
|--------|-------------|
| 💡 Proposed | Draft stage, open for discussion |
| 📝 Review | Ready for review |
| ✅ Accepted | Approved and ready for implementation |
| ⏳ Implemented | Implementation complete |
| ❌ Rejected | Not proceeding |
| 🗑️ Superseded | Replaced by newer RFC |

---

## RFC Template

```markdown
# RFC-NNNN: Title

## Status
- **Proposed:** YYYY-MM-DD
- **Accepted:** YYYY-MM-DD
- **Last Updated:** YYYY-MM-DD

---

## Overview

Brief description of the proposal.

---

## Motivation

Why is this change needed?

---

## Design

Detailed technical design.

---

## Alternatives Considered

What other approaches were evaluated?

---

## Implementation Plan

Step-by-step implementation tasks.

---

## Open Questions

Unresolved issues or decisions.
```

---

## Related Documents

- [Product Specifications](../product/) — Requirements
- [API Specifications](../api/) — Interface definitions
