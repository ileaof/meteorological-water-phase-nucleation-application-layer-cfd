# outputs/

Generated run outputs. The naming convention is:

```
outputs/<scenario>/<run-id>/
```

Each run should save: resolved configuration, software version / git commit,
summary JSON, numerical diagnostics, scientific results, figures, warnings and
validity information.

## What is tracked vs. ignored

- The **flat files** directly under `outputs/` (e.g. `single_state_report.json`)
  are **committed historical reference outputs** from the pre-reorganization
  repository. They are kept tracked as regression references and are **not
  overwritten** by the examples (the examples now write to per-scenario
  subdirectories, `outputs/<scenario>/`).
- **Per-scenario subdirectories** (`outputs/*/`) are gitignored — they hold
  regenerable results and must not be re-committed by accident.

## Regenerating

```bash
python scripts/regenerate_outputs.py     # runs all examples -> outputs/<scenario>/
```