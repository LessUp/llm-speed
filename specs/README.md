---
layout: page
title: Specifications
description: Spec-Driven Development documentation for LLM-Speed
lang: en
---

# Specifications Directory

This directory contains all specification documents following the **Spec-Driven Development (SDD)** paradigm.

---

## Directory Structure

```
specs/
├── product/      # Product requirements and acceptance criteria
├── rfc/          # Technical design documents (Request for Comments)
├── api/          # API interface definitions
├── db/           # Database/schema specifications
└── testing/      # BDD test case specifications
```

---

## Purpose

| Directory | Purpose | When to Update |
|-----------|---------|----------------|
| `product/` | Define *what* to build and *why* | New features, requirement changes |
| `rfc/` | Define *how* to implement | Architecture decisions, technical changes |
| `api/` | Define interfaces | API changes, new functions |
| `db/` | Define data schemas | Schema changes, migrations |
| `testing/` | Define test scenarios | New test cases, edge cases |

---

## Current Specifications

### Product Requirements

| Spec | Description |
|------|-------------|
| [CUDA LLM Kernel Optimization](product/cuda-llm-kernel-optimization.md) | Core feature requirements |

### RFC Documents

| RFC | Status | Title |
|-----|--------|-------|
| [0001](rfc/0001-core-architecture.md) | ✅ Accepted | Core Architecture |
| [0002](rfc/0002-implementation-tasks.md) | ✅ Accepted | Implementation Tasks |

---

## SDD Workflow

1. **Review Specs First** — Read relevant specs before writing code
2. **Update Spec First** — If changing interfaces/features, update specs before coding
3. **Implement Against Spec** — Code must 100% comply with spec definitions
4. **Test Against Spec** — Tests must cover all acceptance criteria

For detailed AI agent workflow instructions, see [`AGENTS.md`](../AGENTS.md).

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-17 | Initial SDD structure setup |
