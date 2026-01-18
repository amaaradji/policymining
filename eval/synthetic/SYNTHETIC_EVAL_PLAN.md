# Synthetic Evaluation Plan (Policy-only conformance) for policymining

## Purpose
We need a correctness-oriented evaluation that addresses the "no ground truth" reviewer concern by:
1) generating synthetic logs,
2) injecting known policy violations (ground truth),
3) running the policy-only replay checker,
4) reporting precision/recall/F1 and breakdowns.

The BPIC 2017 evaluation remains as feasibility only.

---

## High-level pipeline

```mermaid
flowchart LR
  A[Model builder] --> B[PM4Py simulation]
  B --> C[Clean synthetic event log]
  C --> D[Violation injector]
  D --> E[Event log with injected cases]
  D --> F[Ground truth labels]
  E --> G[Policy-only checker\n(reuse existing logic)]
  G --> H[Predictions]
  F --> I[Scoring]
  H --> I
  I --> J[Metrics + plots + tables]

## Quick checkpoint index
- CP0: Bootstrap + reproducibility
- CP1: Clean synthetic log generation
- CP2: Violation injection + ground truth
- CP3: Run policy-only checker
- CP4: Metrics + plots
- CP5: Parameter sweeps
