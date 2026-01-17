# Policy-Only Conformance Checking Framework

This repository implements **policy-only conformance checking** for business process mining—evaluating binding policy violations independently of process model alignment. This work serves as a foundation for future **policy-aware conformance checking** that will integrate policy evaluation with control-flow alignment.

## Overview

Traditional conformance checking focuses on control-flow deviations by comparing observed execution traces with process models. This framework extends conformance checking by introducing **policy logs** to detect binding policy violations—cases where resource assignments, authorizations, or duty obligations are not satisfied.

### Policy-Only vs. Policy-Aware

- **Policy-Only (This Work):** Evaluates binding policy compliance independently of process models
- **Policy-Aware (Future Work):** Integrates policy evaluation with control-flow alignment simultaneously

## Current Implementation: P1 Senior Approval Duty

The framework currently implements **P1**, a senior approval duty policy for high-value loan applications:

**Policy Rule:** If loan amount ≥ T (threshold), then senior approval is *required* within Δ hours (delegation window) of application submission. If no senior approval occurs within Δ, junior approval is *permitted* as a delegation fallback. Otherwise, the duty is *unmet* (violation).

**Parameters:**
- T = 25,000 (amount threshold)
- Δ = 350 hours (~14.5 days, delegation window)

**Outcomes:**
- `not_applicable`: Amount < T (policy does not apply)
- `duty_met`: Senior approval within Δ hours
- `duty_met_via_delegation`: Junior approval (no senior within Δ)
- `duty_unmet`: No approval at all (violation)

## Repository Structure

```
policymining/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
│
├── data/                              # Event log datasets
│   ├── BPI Challenge 2017.xes        # BPI 2017 dataset (31,509 cases)
│   └── BPI Challenge 2017.cached.csv # CSV cache for fast loading
│
├── eval/policy_only/                  # Policy-only evaluation (current work)
│   ├── config.yaml                   # Policy configuration (T, Δ, activities)
│   ├── utils_io.py                   # Event log loading, hybrid role assignment
│   ├── generate_policy_log_p1.py     # Policy log generation with P1 logic
│   ├── run_experiments.py            # Experiments E1-E4
│   ├── policy_log.csv                # Generated policy log (2,748 evaluations)
│   ├── summary.json                  # Outcome statistics
│   ├── experiment_results.json       # Full experimental results
│   ├── results_table.csv             # Parameter sensitivity results
│   ├── fig_outcome_bars.png          # Outcome distribution visualization
│   └── fig_violation_heatmap.png     # Parameter sensitivity heatmap
│
├── experiments/                       # Experimental scripts
│   ├── run_experiments.py            # Experiment orchestration
│   ├── analyze_results.py            # Results analysis
│   └── experiment_run.log            # Execution log
│
└── policy_engine/                     # Policy engine (legacy/other policies)
    └── ...
```

## Key Results (Δ=350h, T=25K)

**Dataset:** BPI Challenge 2017 (5,000 cases sampled → 2,748 policy evaluations)

**Role Assignment:** Hybrid approach identified 27 senior resources (28%) and 70 junior (72%) from 97 total approvers

**Outcome Distribution:**

| Outcome | Count | Overall % | Among Applicable (N=583) |
|---------|-------|-----------|--------------------------|
| Not applicable | 2,165 | 78.8% | - |
| Duty met (senior) | 500 | 18.2% | 85.8% |
| Delegation (junior) | 16 | 0.6% | 2.7% |
| Violation (no approval) | 67 | 2.4% | 11.5% |

**Key Finding:** 67 cases (2.4%) flagged as violations completed their control flow successfully—these would be **invisible to traditional control-flow conformance checking**.

## Features

### Hybrid Role Assignment
When organizational metadata is unavailable, the framework infers senior/junior roles using:
1. Regex pattern matching (`(?i)(SENIOR|MANAGER)`)
2. Activity-based patterns (fallback):
   - Top 30% most active approvers → senior
   - High-value case handlers (>$50K) → senior

### Delegation Detection
Enhanced logic detects three delegation scenarios:
1. **In-window delegation:** Junior approves within Δ, no senior present
2. **Late delegation:** Junior approves beyond Δ, no senior ever
3. **No approval:** Neither senior nor junior → violation

### Replay-Based Evaluation
For each target event (O_Accepted, A_Approved), the system replays case history to verify:
- Does amount trigger policy? (≥ T)
- Did senior approve within Δ?
- If not, did junior approve?
- Outcome: duty_met, delegation, or violation

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/policymining.git
cd policymining

# Install dependencies
pip install pandas pm4py pyyaml matplotlib seaborn

# Download BPI 2017 dataset
# Place in data/BPI Challenge 2017.xes
```

## Usage

### Generate Policy Log

```bash
cd eval/policy_only
python generate_policy_log_p1.py --config config.yaml --max-cases 5000
```

**Outputs:**
- `policy_log.csv` - Policy evaluations (2,748 rows × 16 columns)
- `summary.json` - Outcome statistics

### Run Experiments

```bash
python run_experiments.py --policy-log policy_log.csv --config config.yaml
```

**Outputs:**
- `experiment_results.json` - Full experimental data
- `results_table.csv` - Parameter sensitivity results
- `fig_outcome_bars.png` - Outcome distribution
- `fig_violation_heatmap.png` - Parameter sensitivity

### Configuration

Edit `config.yaml` to customize:

```yaml
policy_config:
  T: 25000              # Amount threshold
  delta_hours: 350      # Delegation window (hours)
  request_anchors:      # Activities marking request start
    - "A_Submitted"
  approval_acts:        # Activities representing approvals
    - "W_Validate application"
    - "W_Approve offer"
  target_acts:          # Events where policy is evaluated
    - "O_Accepted"
    - "A_Approved"
  senior_regex: "(?i)(SENIOR|MANAGER)"  # Senior identification pattern
```

## Experiments

### E1: Outcome Distribution
Reports counts and rates for each policy outcome (not_applicable, duty_met, delegation, violation)

### E2: Policy Value
Measures percentage of cases flagged as violations that completed control flow successfully (invisible to traditional conformance checking)

### E3: Sensitivity Analysis (Optional)
Tests parameter variations to understand T and Δ sensitivity

### E4: Early Warning (Optional)
Simulates real-time violation detection based on delegation cases

## Parameter Calibration

**Delegation Window (Δ) Sensitivity:**

| Δ (hours) | Duty Met | Delegation | Violations | Assessment |
|-----------|----------|------------|------------|------------|
| 200h | 13.9% | 0.04% | 7.3% | Too many violations |
| **350h** | **18.2%** | **0.6%** | **2.4%** | **Balanced ✓** |
| 500h | 19.8% | 0.04% | 1.3% | No delegation visible |

**Lesson:** Δ must be positioned between Q1 (265h) and median (341h) wait times to capture realistic delegation scenarios.

## Limitations

1. **No ground truth** - Cannot assess precision/recall without validated violation set
2. **Inferred roles** - Senior assignments based on behavioral patterns, not organizational hierarchy
3. **Single policy** - Only P1 tested (senior approval duty)
4. **Data quality** - Potential false positives in flagged violations
5. **Policy-only scope** - Not integrated with process models (policy-aware is future work)

## Future Work: Policy-Aware Conformance Checking

The next phase will integrate policy logs with process model alignment:

1. **Incorporate organizational metadata** from external sources
2. **Align execution traces** with process models AND binding policies simultaneously
3. **Validate violations** against ground-truth labels or domain expert judgments
4. **Test additional policies** (prohibitions, permissions, cardinality constraints)
5. **Evaluate generalizability** on additional datasets



## Acknowledgments

- BPI Challenge 2017 dataset from Eindhoven University of Technology
- PM4Py library for process mining operations
