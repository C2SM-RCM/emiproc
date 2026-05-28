# Emiproc Project Guidelines

## Scope
These instructions apply to all emiproc tasks in this repository.

## Mandatory First Read (All Agents And Skills)
- Read [AI_README.md](./AI_README.md) before any analysis, plan, code edit, or review.
- Treat the scientific guardrails and validation checklist in that file as required constraints.

## Architecture
- Core package code is in [emiproc](../emiproc).
- Inventory readers and adapters are in [emiproc/inventories](../emiproc/inventories).
- Export logic is in [emiproc/exports](../emiproc/exports).
- Integration and behavior checks are in [tests](../tests).
- User-facing processing examples are in [scripts](../scripts) and [docs/source/tutos](../docs/source/tutos).

## Documentation Expectations
If behavior or interfaces visible to users change, update relevant docs or tutorial references in [docs/source](../docs/source).
