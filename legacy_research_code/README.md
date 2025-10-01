# Legacy Research Code

This directory contains the original implementations used for research experiments in the paper "Complementing Event Logs with Policy Logs for Business Process Mining."

## ⚠️ Note

**These implementations are preserved for research reproducibility only.**

For new projects, please use the unified policy engine in [`../policy_engine/`](../policy_engine/).

## Contents

- **`resource_availability_policy.py`**: Original implementation focusing on resource availability constraints with visualization and comprehensive evaluation
- **`resource_availability_policy_fixed.py`**: Fixed version with bug corrections
- **`policy_mining.py`**: Early policy mining utilities
- **`policy_mining_optimized.py`**: Performance-optimized version

## Research Features

These implementations include features specific to the original research:

- **Visualization**: matplotlib-based plots for violation detection results
- **Event-log-only comparison**: Statistical inference approach for baseline comparison
- **XES policy log generation**: pm4py-based policy log export
- **Comprehensive evaluation**: Ground truth testing with GT1/GT2 approaches

## Migration to Unified Framework

The functionality from these implementations has been consolidated into the unified policy engine:

| Legacy Feature | Unified Framework Location |
|----------------|---------------------------|
| Resource availability checking | `policy_engine.py` → `ResourceAvailabilityPolicy` (P2) |
| Policy evaluation | `policy_engine.py` → `PolicyEngine` |
| Synthetic violation injection | `evaluation.py` → `PolicyEvaluator` |
| Performance metrics | `evaluation.py` → `evaluate_detection()` |

## When to Use Legacy Code

Use these implementations only if you need to:
- Reproduce exact results from the original paper
- Access specific visualization code
- Reference the event-log-only detection baseline

For all other purposes, use the unified policy engine.
