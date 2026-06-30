# Examples

This directory contains small synthetic datasets for testing and learning the `nmr_bind_fit` workflow.

| File | Intended signal |
|------|-----------------|
| `synthetic_11.csv` | 1:1 binding |
| `synthetic_12.csv` | sequential 1:2 binding |
| `synthetic_21.csv` | sequential 2:1 binding |
| `synthetic_nonbinding.csv` | non-binding linear drift |

Run one example with:

```bash
nmr_bind_fit --input examples/synthetic_11.csv --bootstrap 200 --seed 1
```

The datasets are deterministic and are not experimental measurements. They are intended for smoke tests, documentation, tutorials, and JOSS-readiness examples. Use real experimental data only when you have permission to share it publicly.

Regenerate the CSV files with:

```bash
python examples/generate_synthetic_examples.py
```
