---
name: Emiproc Maintainer Review
description: "Use when reviewing emiproc changes for regressions, scientific consistency, unit/category integrity, missing tests, and docs impact before merge."
argument-hint: "scope=<files or feature>"
agent: "agent"
---
Perform a maintainer-grade review of the requested emiproc changes.

Before starting the review, read [.github/AI_README.md](../AI_README.md) and apply its scientific guardrails and validation expectations.

Review priorities, highest to lowest:
1. Bugs and behavioral regressions.
2. Scientific consistency risks (units, totals, category mappings, speciation logic, CRS usage).
3. API compatibility and downstream break risk.
4. Missing or weak tests for changed behavior.
5. Documentation or tutorial update gaps.

Expected output format:
- Findings (ordered by severity): include file path and precise line references.
- Open questions and assumptions.
- Suggested additional tests.
- Brief change summary.

Review requirements:
- Prefer concrete evidence from changed code and nearby tests.
- Flag silent behavior changes explicitly.
- If no findings are detected, state that and mention residual risk areas.
