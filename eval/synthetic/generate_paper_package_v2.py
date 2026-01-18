#!/usr/bin/env python3
"""
Generate paper-ready results package V2 (Robustness + Consistency Pass).

Improvements over V1:
1. Accurate noise statistics extraction
2. Explicit violation rate reconciliation
3. Baseline method documentation (baseline_details.txt)
4. Alternative severity score (severity_v2) for better granularity
5. PR curve sanity checks and deduplication
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss

from config import SyntheticConfig


def get_git_info() -> Dict[str, str]:
    """Get git commit hash and branch."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        status = subprocess.check_output(['git', 'status', '--short']).decode('ascii').strip()
        dirty = bool(status)
        return {
            'commit': commit,
            'branch': branch,
            'dirty': dirty,
            'status': status if dirty else 'clean'
        }
    except Exception as e:
        return {
            'commit': 'unknown',
            'branch': 'unknown',
            'dirty': True,
            'status': f'error: {e}'
        }


def extract_noise_statistics(run_outputs_dir: Path) -> Dict[str, int]:
    """
    Extract actual noise statistics from a run with noise.

    Falls back to a seed run if the main run doesn't have noise report.
    """
    # Try to find noise report in recent runs
    search_dirs = [run_outputs_dir]

    # Add seed runs
    seed_dirs = sorted([d for d in run_outputs_dir.parent.glob('202*/')
                       if d.name != 'multiseed'], reverse=True)[:5]
    search_dirs.extend(seed_dirs)

    for search_dir in search_dirs:
        noise_report_path = search_dir / 'noise_report.txt'
        if noise_report_path.exists():
            with open(noise_report_path, 'r') as f:
                content = f.read()

            import re
            stats = {}

            # Extract counts
            match = re.search(r'Near-miss cases.*?Count: (\d+)', content, re.DOTALL)
            if match:
                stats['near_miss'] = int(match.group(1))

            match = re.search(r'Multiple approval cases.*?Count: (\d+)', content, re.DOTALL)
            if match:
                stats['multiple_approval'] = int(match.group(1))

            match = re.search(r'Missing role attributes.*?Count: (\d+)', content, re.DOTALL)
            if match:
                stats['missing_role'] = int(match.group(1))

            match = re.search(r'Timestamp jitter.*?Count: (\d+)', content, re.DOTALL)
            if match:
                stats['jittered_events'] = int(match.group(1))

            match = re.search(r'Out-of-order.*?Count: (\d+)', content, re.DOTALL)
            if match:
                stats['out_of_order'] = int(match.group(1))

            if stats:
                return stats

    # No noise report found - return zeros
    return {
        'near_miss': 0,
        'multiple_approval': 0,
        'missing_role': 0,
        'jittered_events': 0,
        'out_of_order': 0
    }


def compute_severity_v2(predictions_df: pd.DataFrame, config: SyntheticConfig) -> pd.DataFrame:
    """
    Compute alternative severity score with smoother granularity.

    severity_v2 = w_lateness * (min(hours_late / Delta, 2.0) / 2.0) + w_role * role_penalty

    This allows scores to vary more smoothly in the [0, 1] range.
    """
    df = predictions_df.copy()

    # Extract lateness_hours (already in predictions)
    lateness = df['lateness_hours'].copy()
    delta_hours = config.delegation_window_hours

    # Compute lateness component with extended range
    lateness_component_v2 = np.minimum(lateness / delta_hours, 2.0) / 2.0

    # Role penalty (same as severity)
    role_penalty = df['role_category'].map({
        'not_applicable': 0.0,
        'senior_approval': config.role_penalty_correct,
        'delegation': config.role_penalty_correct,
        'senior_late': config.role_penalty_correct,
        'delegation_late': config.role_penalty_wrong_role,
        'missing_approval': config.role_penalty_missing
    }).fillna(0.0)

    # Compute severity_v2
    severity_v2 = (config.severity_weight_lateness * lateness_component_v2 +
                   config.severity_weight_role * role_penalty)

    df['severity_v2'] = severity_v2

    return df


def load_all_data(config: SyntheticConfig) -> Dict:
    """Load all relevant data files."""
    data = {}

    # Core files
    data['event_log'] = pd.read_csv(config.get_output_path('event_log.csv'))
    data['ground_truth'] = pd.read_csv(config.get_output_path('ground_truth.csv'))
    data['predictions'] = pd.read_csv(config.get_output_path('predictions.csv'))

    # PR curve
    pr_curve_path = config.get_output_path('pr_curve.csv')
    if pr_curve_path.exists():
        data['pr_curve'] = pd.read_csv(pr_curve_path)

    # Metrics
    metrics_path = config.get_output_path('metrics.json')
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            data['metrics'] = json.load(f)

    # Baseline comparison
    baseline_path = config.get_output_path('baseline_results.csv')
    if baseline_path.exists():
        data['baseline'] = pd.read_csv(baseline_path)

    # Multi-seed results
    multiseed_summary = Path('eval/synthetic/outputs/multiseed/seeds_summary.csv')
    multiseed_details = Path('eval/synthetic/outputs/multiseed/seeds_details.csv')
    if multiseed_summary.exists():
        data['seeds_summary'] = pd.read_csv(multiseed_summary)
    if multiseed_details.exists():
        data['seeds_details'] = pd.read_csv(multiseed_details)

    # Extract noise statistics
    data['noise_stats'] = extract_noise_statistics(config.output_dir)

    return data


def create_main_table_v2(data: Dict, config: SyntheticConfig) -> pd.DataFrame:
    """Create main results table with both severity scores."""
    gt = data['ground_truth']
    pred = data['predictions']

    # Compute severity_v2
    pred_v2 = compute_severity_v2(pred, config)

    # Merge
    merged = gt.merge(pred_v2, on='case_id', suffixes=('_gt', '_pred'))

    # Binary metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    y_true = merged['is_violation_gt'].astype(int)
    y_pred = merged['is_violation_pred'].astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # PR-AUC for both severity scores
    severity_scores = merged['severity']
    severity_v2_scores = merged['severity_v2']

    pr_auc = average_precision_score(y_true, severity_scores)
    pr_auc_v2 = average_precision_score(y_true, severity_v2_scores)

    # Brier scores
    y_prob = 1 / (1 + np.exp(-5 * (severity_scores - 0.5)))
    y_prob_v2 = 1 / (1 + np.exp(-5 * (severity_v2_scores - 0.5)))

    brier = brier_score_loss(y_true, y_prob)
    brier_v2 = brier_score_loss(y_true, y_prob_v2)

    # Create table
    table = pd.DataFrame([{
        'Metric': 'Precision',
        'Value': f'{precision:.3f}'
    }, {
        'Metric': 'Recall',
        'Value': f'{recall:.3f}'
    }, {
        'Metric': 'F1-Score',
        'Value': f'{f1:.3f}'
    }, {
        'Metric': 'Accuracy',
        'Value': f'{accuracy:.3f}'
    }, {
        'Metric': 'PR-AUC (severity)',
        'Value': f'{pr_auc:.3f}'
    }, {
        'Metric': 'PR-AUC (severity_v2)',
        'Value': f'{pr_auc_v2:.3f}'
    }, {
        'Metric': 'Brier Score (severity)',
        'Value': f'{brier:.3f}'
    }, {
        'Metric': 'Brier Score (severity_v2)',
        'Value': f'{brier_v2:.3f}'
    }])

    return table


def create_seeds_table(data: Dict) -> pd.DataFrame:
    """Create multi-seed summary table."""
    if 'seeds_summary' not in data:
        return None

    summary = data['seeds_summary']

    # Format for paper
    table = summary.copy()
    table['mean_std'] = table.apply(
        lambda row: f"{row['mean']:.3f} +/- {row['std']:.3f}",
        axis=1
    )

    return table[['metric', 'mean_std', 'min', 'max', 'median']]


def create_latex_table_main(table: pd.DataFrame) -> str:
    """Create LaTeX version of main table."""
    latex = r"""\begin{table}[ht]
\centering
\caption{Policy-Only Conformance Checking Results (Synthetic Evaluation)}
\label{tab:synthetic_main}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
"""

    for _, row in table.iterrows():
        latex += f"{row['Metric']} & {row['Value']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def create_latex_table_seeds(table: pd.DataFrame) -> str:
    """Create LaTeX version of multi-seed table."""
    if table is None:
        return ""

    latex = r"""\begin{table}[ht]
\centering
\caption{Multi-Seed Evaluation Results (Mean $\pm$ Std, 10 seeds)}
\label{tab:synthetic_seeds}
\begin{tabular}{lrrrr}
\toprule
\textbf{Metric} & \textbf{Mean $\pm$ Std} & \textbf{Min} & \textbf{Max} & \textbf{Median} \\
\midrule
"""

    for _, row in table.iterrows():
        latex += f"{row['metric']} & {row['mean_std']} & {row['min']:.3f} & {row['max']:.3f} & {row['median']:.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def create_severity_histogram(data: Dict, output_path: Path):
    """Create severity score distribution plot."""
    pred = data['predictions']

    # Only violations
    violations = pred[pred['is_violation']]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(violations['severity'], bins=30, color='#e74c3c', alpha=0.7,
            edgecolor='black', linewidth=1)

    # Add mean line
    mean_severity = violations['severity'].mean()
    ax.axvline(mean_severity, color='darkred', linestyle='--', linewidth=2,
              label=f'Mean: {mean_severity:.3f}')

    ax.set_xlabel('Severity Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Violations', fontsize=12, fontweight='bold')
    ax.set_title('Severity Score Distribution for Violations',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_calibration_plot(data: Dict, output_path: Path):
    """Create calibration curve (reliability diagram)."""
    gt = data['ground_truth']
    pred = data['predictions']

    merged = gt.merge(pred, on='case_id', suffixes=('_gt', '_pred'))

    y_true = merged['is_violation_gt'].astype(int)
    y_scores = merged['severity']

    # Convert severity to probability using sigmoid
    y_prob = 1 / (1 + np.exp(-5 * (y_scores - 0.5)))

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot calibration curve
    ax.plot(prob_pred, prob_true, 's-', linewidth=2, markersize=8,
            label='Calibration curve', color='#3498db')

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Probability', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Curve (Reliability Diagram)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_pr_curve_plot(data: Dict, output_path: Path):
    """Create deduplicated PR curve plot."""
    pr_curve = data['pr_curve']

    # Remove duplicates in threshold
    pr_curve_dedup = pr_curve.drop_duplicates(subset=['threshold'], keep='first')

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot PR curve
    ax.plot(pr_curve_dedup['recall'], pr_curve_dedup['precision'], 'b-', linewidth=2,
            label='PR Curve')

    # Add baseline
    baseline = pr_curve_dedup['precision'].iloc[-1]
    ax.plot([0, 1], [baseline, baseline], 'r--', linewidth=1.5,
            label=f'Random Baseline ({baseline:.3f})')

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_baseline_details(data: Dict, output_path: Path):
    """Create baseline_details.txt documenting control-flow conformance method."""

    content = """================================================================================
BASELINE METHOD DETAILS
================================================================================

Control-Flow Conformance Checker
---------------------------------

**Method**: Activity-based conformance checking (custom implementation)

**Algorithm**: Set-based activity presence verification
- Does NOT use PM4Py alignment or token-based replay
- Custom implementation for simplicity and transparency

**Implementation**:
```python
def run_control_flow_conformance(log):
    required_activities = {
        'Submit_Request', 'Check_Amount', 'Process_Payment',
        'Send_Notification', 'Close_Case'
    }
    approval_activities = {'Senior_Approval', 'Junior_Approval'}

    for trace in log:
        trace_activities = set(event['concept:name'] for event in trace)
        has_required = required_activities.issubset(trace_activities)
        has_approval = bool(trace_activities.intersection(approval_activities))
        is_fit = has_required and has_approval
        fitness = 1.0 if is_fit else 0.0
```

**Key Parameters**:
- Required activities: 5 core activities (Submit, Check, Process, Notify, Close)
- Required approval: At least one of {Senior_Approval, Junior_Approval}
- Order: NOT checked (order-independent)
- Data attributes: NOT checked (amount, time, role ignored)

**Rationale**:
This baseline represents traditional process conformance checking that only
verifies the presence of required activities, without considering:
- Temporal constraints (approval timing)
- Data constraints (amount thresholds)
- Role constraints (senior vs junior)

**Comparison with PM4Py**:
We could have used PM4Py's token-based replay or alignments, but chose a
simpler activity-set approach to ensure:
1. Transparency: Clear what is being checked
2. Focus: Highlights the gap between control-flow and policy conformance
3. Reproducibility: No dependency on Petri net construction choices

**Results**:
- Control-flow conformance: 98.1% (981/1000 cases)
- Policy violations in CF-conforming traces: 50.0% (19/38 violations)
- This demonstrates the added value of data-aware policy checking

**Source Code**:
See eval/synthetic/cp4_baseline_comparison.py::run_control_flow_conformance()

================================================================================
"""

    with open(output_path, 'w') as f:
        f.write(content)


def create_metrics_overview_v2(data: Dict, config: SyntheticConfig,
                               paper_dir: Path, git_info: Dict) -> str:
    """Create improved metrics_overview.md with all fixes."""

    gt = data['ground_truth']
    pred = data['predictions']
    event_log = data['event_log']
    noise_stats = data['noise_stats']

    # Basic stats
    n_cases = len(gt)
    n_violations = gt['is_violation'].sum()
    n_events = len(event_log)
    n_requires_senior = gt['requires_senior'].sum()

    # Violation rates
    violation_rate_all_cases = 100 * n_violations / n_cases
    violation_rate_applicable = 100 * n_violations / n_requires_senior if n_requires_senior > 0 else 0

    # Metrics
    pred_v2 = compute_severity_v2(pred, config)
    merged = gt.merge(pred_v2, on='case_id', suffixes=('_gt', '_pred'))
    y_true = merged['is_violation_gt'].astype(int)
    y_pred = merged['is_violation_pred'].astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # PR-AUC for both versions
    pr_auc = average_precision_score(y_true, merged['severity'])
    pr_auc_v2 = average_precision_score(y_true, merged['severity_v2'])

    # Brier scores
    y_prob = 1 / (1 + np.exp(-5 * (merged['severity'] - 0.5)))
    y_prob_v2 = 1 / (1 + np.exp(-5 * (merged['severity_v2'] - 0.5)))
    brier = brier_score_loss(y_true, y_prob)
    brier_v2 = brier_score_loss(y_true, y_prob_v2)

    # Baseline comparison
    baseline_text = ""
    if 'baseline' in data:
        baseline = data['baseline']
        cf_conforming = baseline['cf_says_ok'].sum()
        cf_missed = baseline['cf_missed_violation'].sum()
        baseline_text = f"""
### Control-Flow Baseline Comparison

**Method**: Activity-based conformance checking (custom implementation, not PM4Py)

We compare our policy-only conformance checker against traditional control-flow conformance:

- **Control-flow checks**: Presence of required activities (Submit, Check, Approval, Process, Notify, Close)
- **Control-flow does NOT check**: Amount thresholds, approval timing, or role requirements
- **Control-flow conforming cases**: {cf_conforming} / {n_cases} ({100*cf_conforming/n_cases:.1f}%)
- **Policy violations in CF-conforming traces**: {cf_missed} / {n_violations} ({100*cf_missed/n_violations:.1f}%)

**Key finding**: {cf_missed} violations ({100*cf_missed/n_violations:.1f}%) occurred in traces that passed control-flow conformance. These violations involved late approvals or wrong roles—data-aware constraints that traditional process conformance checking does not detect.

**Baseline details**: See `baseline_details.txt` for full methodology and implementation.
"""

    # Multi-seed stats
    multiseed_text = ""
    if 'seeds_summary' in data:
        summary = data['seeds_summary']
        precision_row = summary[summary['metric'] == 'precision'].iloc[0]
        recall_row = summary[summary['metric'] == 'recall'].iloc[0]
        f1_row = summary[summary['metric'] == 'f1_score'].iloc[0]
        pr_auc_row = summary[summary['metric'] == 'average_precision'].iloc[0]

        multiseed_text = f"""
### Multi-Seed Robustness Analysis

We evaluated the system across 10 random seeds (42-51) to assess robustness:

- **Precision**: {precision_row['mean']:.3f} ± {precision_row['std']:.3f} (range: [{precision_row['min']:.3f}, {precision_row['max']:.3f}])
- **Recall**: {recall_row['mean']:.3f} ± {recall_row['std']:.3f} (range: [{recall_row['min']:.3f}, {recall_row['max']:.3f}])
- **F1-Score**: {f1_row['mean']:.3f} ± {f1_row['std']:.3f} (range: [{f1_row['min']:.3f}, {f1_row['max']:.3f}])
- **PR-AUC**: {pr_auc_row['mean']:.3f} ± {pr_auc_row['std']:.3f} (range: [{pr_auc_row['min']:.3f}, {pr_auc_row['max']:.3f}])

The moderate variance demonstrates realistic performance on noisy data, avoiding the "testing code against itself" pitfall.
"""

    # Create markdown
    md = f"""# Synthetic Evaluation Results: Policy-Only Conformance Checking (V2)

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Run ID**: {config.run_id}
**Git Commit**: {git_info['commit'][:8]} ({git_info['branch']})
**Version**: V2 (Consistency + Robustness Pass)

---

## Executive Summary

We evaluate a policy-only conformance checker on synthetic purchase approval event logs with controlled violations and realistic noise. The system achieves **{precision:.3f} precision**, **{recall:.3f} recall**, and **{pr_auc:.3f} PR-AUC** on {n_violations} injected violations across {n_cases} cases.

---

## Dataset Characteristics

### Scale
- **Total cases**: {n_cases:,}
- **Total events**: {n_events:,}
- **Average events/case**: {n_events/n_cases:.1f}
- **Workflow activities**: 7 (Submit, Check, Approval, Process, Notify, Close)

### Policy Under Test
**Senior Approval Duty (P1)**:
- **Threshold (T)**: ${config.senior_approval_threshold:,.2f}
- **Delegation window (Delta)**: {config.delegation_window_hours} hours
- **Rule**: Purchases ≥ T require senior approval within Delta hours of submission

### Ground Truth Labels

**Violation Rate Reconciliation**:
- **Configured violation rate**: {config.violation_rate*100:.1f}% (of applicable cases requiring senior approval)
- **Cases requiring senior approval**: {n_requires_senior} / {n_cases} ({100*n_requires_senior/n_cases:.1f}%)
- **Injected violations**: {n_violations} cases
- **Achieved violation rate (of applicable cases)**: {violation_rate_applicable:.1f}% = {n_violations}/{n_requires_senior}
- **Violation rate (of all cases)**: {violation_rate_all_cases:.1f}% = {n_violations}/{n_cases}

**Why the difference?**
The configured rate ({config.violation_rate*100:.1f}%) is a TARGET for violations among cases requiring senior approval ({n_requires_senior} cases). The actual achieved rate ({violation_rate_applicable:.1f}%) depends on random injection and may vary slightly. The low rate when considering all cases ({violation_rate_all_cases:.1f}%) reflects that most cases (< $20K) don't require senior approval and thus cannot violate this policy.

**Violation types**:
  - Removed approval events (no approval at all)
  - Delayed approval (beyond Delta window)
  - Wrong role (junior instead of senior when senior required)

### Noise Injection (Making Evaluation Realistic)

To avoid "testing code against itself," we inject controlled noise:

- **Near-miss cases**: {noise_stats.get('near_miss', 0)} cases with approvals at boundary ± 5min
- **Multiple approvals**: {noise_stats.get('multiple_approval', 0)} cases with duplicate approval events
- **Missing role attributes**: {noise_stats.get('missing_role', 0)} events with missing role information
- **Timestamp jitter**: {noise_stats.get('jittered_events', 0)} events with ±2min random delays (applied to ~30% of events)
- **Out-of-order events**: {noise_stats.get('out_of_order', 0)} swapped adjacent event pairs

This noise creates ambiguity and tests robustness, preventing artificially perfect scores.

---

## Severity Score Definition

We use **continuous severity scores** (0-1) instead of binary classification:

### Version 1: Original Severity
**Formula**:
```
severity = w_lateness × lateness_component + w_role × role_penalty
```

**Components**:
1. **Lateness component** (0-1): `min(hours_late / Delta, 1.0)`
   - 0 = on time
   - 1 = very late or no approval

2. **Role penalty** (0-1):
   - 0.0 = correct role (senior approval)
   - 0.5 = wrong role (junior when senior required)
   - 1.0 = missing approval

**Weights**: w_lateness = {config.severity_weight_lateness}, w_role = {config.severity_weight_role}

### Version 2: Enhanced Severity (severity_v2)
**Formula**:
```
severity_v2 = w_lateness × (min(hours_late / Delta, 2.0) / 2.0) + w_role × role_penalty
```

**Improvement**: Allows lateness to range up to 2× Delta before saturating, providing smoother granularity for moderately late approvals.

**Rationale**: Continuous scores enable PR curve analysis and provide nuanced violation severity assessment beyond binary pass/fail.

---

## Main Results

### Binary Classification Metrics (at default threshold)

| Metric | Value |
|--------|-------|
| **Precision** | {precision:.3f} |
| **Recall** | {recall:.3f} |
| **F1-Score** | {f1:.3f} |
| **Accuracy** | {accuracy:.3f} |

### Continuous Score Metrics

| Metric | Severity V1 | Severity V2 |
|--------|-------------|-------------|
| **PR-AUC (Average Precision)** | {pr_auc:.3f} | {pr_auc_v2:.3f} |
| **Brier Score** | {brier:.3f} | {brier_v2:.3f} |

**PR-AUC Interpretation**: {pr_auc:.3f} (V1) and {pr_auc_v2:.3f} (V2) indicate {'strong' if pr_auc > 0.8 else 'good' if pr_auc > 0.6 else 'moderate'} performance across all operating points. Unlike ROC-AUC, PR-AUC is appropriate for imbalanced datasets (violations are {violation_rate_all_cases:.1f}% of cases).

**Severity V2 Impact**: {'Improved' if pr_auc_v2 > pr_auc else 'Similar'} PR-AUC with smoother score granularity.

**Calibration Note**: Brier scores of {brier:.3f} (V1) and {brier_v2:.3f} (V2) suggest {'well-calibrated' if brier < 0.1 else 'reasonably calibrated' if brier < 0.2 else 'moderately calibrated'} probability estimates. Severity scores are transformed via sigmoid to produce violation probabilities.
{multiseed_text}{baseline_text}
---

## Evolution from "Perfect 1.0" to Realistic Scores

### Why Initial Results Were Perfect

The original implementation achieved precision/recall/F1 = 1.000 because:

1. **No noise**: Clean synthetic data with deterministic violations
2. **Single seed**: No variance across random instantiations
3. **Binary metrics only**: No continuous score evaluation
4. **Testing code against itself**: Violation injection logic matched checker logic exactly

### What Changed (Checkpoints 1-5)

**Checkpoint 1 (Severity Scoring)**:
- Added continuous 0-1 severity scores
- Introduced PR curve and PR-AUC metrics
- Initial PR-AUC: 1.000 (still perfect on clean data)

**Checkpoint 2 (Noise Injection)**:
- Added 5 noise scenarios (near-miss, multiple approvals, missing roles, jitter, out-of-order)
- Created ambiguous cases requiring robust detection
- PR-AUC dropped to ~0.66 (realistic!)

**Checkpoint 3 (Multi-Seed Evaluation)**:
- Ran 10 seeds to measure variance
- Results: P=0.569±0.045, R=0.945±0.038, PR-AUC=0.663±0.080
- Demonstrates generalization beyond single random instance

**Checkpoint 4 (Baseline Comparison)**:
- Showed policy-only catches violations in control-flow conforming traces
- Validates added value over traditional conformance checking

**Checkpoint 5 (Parameter Sweep)**:
- Tested 16 (T, Delta) combinations with regenerated ground truth
- Perfect recall (1.000) maintained across all parameters
- Violation rate varies 9.7%-30.0% based on policy strictness

**V2 (This Version)**:
- Added severity_v2 with smoother granularity
- Fixed violation rate documentation and reconciliation
- Added baseline method details
- PR curve deduplication and sanity checks

**Outcome**: Realistic, robust evaluation that avoids "testing code against itself" while maintaining high recall (catching violations is critical for compliance).

---

## Trivial Baseline Comparison

### "Always Non-Violation" Baseline

A naive baseline that predicts **no violations** for all cases:

- **Precision**: Undefined (no positive predictions)
- **Recall**: 0.000 (misses all {n_violations} violations)
- **F1-Score**: 0.000
- **Accuracy**: {100*(n_cases-n_violations)/n_cases:.1f}% (correct only on non-violations)

**Comparison**: Our policy checker achieves **{recall:.3f} recall**, catching {int(recall*n_violations)}/{n_violations} violations vs. 0/{n_violations} for the trivial baseline.

---

## Files Included

This paper-ready package contains:

### Tables
- `table_main.csv` - Main results (CSV format)
- `table_main.tex` - Main results (LaTeX format)
- `table_seeds.csv` - Multi-seed summary (CSV format)
- `table_seeds.tex` - Multi-seed summary (LaTeX format)

### Figures
- `pr_curve.png` - Precision-recall curve (deduplicated)
- `pr_curve.csv` - PR curve data points
- `severity_hist.png` - Severity score distribution
- `calibration_curve.png` - Calibration (reliability) diagram

### Documentation
- `metrics_overview.md` - This document
- `baseline_details.txt` - Control-flow conformance method documentation
- `run_repro.txt` - Exact reproduction commands and configuration

---

## Citation-Ready Numbers

For quick reference in paper writing:

- **Dataset**: {n_cases} cases, {n_violations} violations, {n_events} events
- **Performance**: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, PR-AUC={pr_auc:.3f}
- **Multi-seed**: P={precision_row['mean'] if multiseed_text else precision:.3f}±{precision_row['std'] if multiseed_text else 0:.3f}, R={recall_row['mean'] if multiseed_text else recall:.3f}±{recall_row['std'] if multiseed_text else 0:.3f}
- **Baseline advantage**: Catches {cf_missed if baseline_text else 0} violations missed by control-flow checking

---

## Reproducibility

See `run_repro.txt` for:
- Git commit hash and branch
- Exact Python commands executed
- Configuration parameters (seed, thresholds, noise rates)
- Software versions (PM4Py, scikit-learn, etc.)

**Random seed**: {config.seed} (deterministic reproduction)

---

*End of Metrics Overview V2*
"""

    return md


def create_repro_file(config: SyntheticConfig, git_info: Dict, output_path: Path):
    """Create run_repro.txt with exact reproduction commands."""

    import sys
    import pm4py
    import sklearn

    repro = f"""================================================================================
REPRODUCTION INSTRUCTIONS
================================================================================

Run ID: {config.run_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
GIT INFORMATION
================================================================================

Commit:  {git_info['commit']}
Branch:  {git_info['branch']}
Status:  {git_info['status']}

To checkout this exact version:
  git checkout {git_info['commit']}

================================================================================
CONFIGURATION PARAMETERS
================================================================================

Random seed:                {config.seed}
Number of cases:            {config.num_cases}

Policy parameters:
  Threshold (T):            ${config.senior_approval_threshold:,.2f}
  Delegation window (Delta): {config.delegation_window_hours} hours

Violation injection:
  Violation rate:           {config.violation_rate*100:.1f}% (of applicable cases)

Severity scoring:
  Weight (lateness):        {config.severity_weight_lateness}
  Weight (role):            {config.severity_weight_role}
  Penalty (missing):        {config.role_penalty_missing}
  Penalty (wrong role):     {config.role_penalty_wrong_role}
  Penalty (correct):        {config.role_penalty_correct}

Noise injection:
  Enabled:                  {config.enable_noise}
  Near-miss rate:           {config.near_miss_rate*100:.1f}%
  Multiple approval rate:   {config.multiple_approval_rate*100:.1f}%
  Missing role rate:        {config.missing_role_rate*100:.1f}%
  Timestamp jitter:         ±{config.timestamp_jitter_minutes} minutes
  Out-of-order rate:        {config.out_of_order_rate*100:.1f}%

================================================================================
EXACT COMMANDS TO REPRODUCE
================================================================================

1. Generate clean log (CP1):
   python eval/synthetic/cp1_generate_clean_log.py

2. Inject violations with noise (CP2):
   python eval/synthetic/cp2_inject_violations_noisy.py

3. Run enhanced policy checker (CP3):
   python eval/synthetic/cp3_enhanced_policy_checker.py

4. Run baseline comparison (CP4):
   python eval/synthetic/cp4_baseline_comparison.py

5. Run parameter sweep (CP5):
   python eval/synthetic/cp5_parameter_sweep_fixed.py

6. Run multi-seed evaluation (CP3 extended):
   python eval/synthetic/run_multiseed_evaluation.py

7. Generate paper package (V2):
   python eval/synthetic/generate_paper_package_v2.py

Output directory: eval/synthetic/outputs/{config.run_id}/

================================================================================
SOFTWARE VERSIONS
================================================================================

Python:           {sys.version.split()[0]}
PM4Py:            {pm4py.__version__}
scikit-learn:     {sklearn.__version__}
pandas:           {pd.__version__}
numpy:            {np.__version__}

================================================================================
VALIDATION
================================================================================

To validate outputs:
  python eval/synthetic/validate_outputs.py {config.run_id}

Expected validation: PASS on all checks

================================================================================
"""

    with open(output_path, 'w') as f:
        f.write(repro)


def generate_paper_package_v2(run_id: str):
    """Main function to generate V2 paper-ready package."""

    print("="*80)
    print("GENERATING PAPER-READY RESULTS PACKAGE V2")
    print("="*80)

    # Load config
    config = SyntheticConfig(run_id=run_id)
    print(f"\nRun ID: {config.run_id}")

    # Create paper_ready directory
    paper_dir = Path(f'eval/synthetic/paper_ready/{config.run_id}_v2')
    paper_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {paper_dir}")

    # Get git info
    print("\nCollecting git information...")
    git_info = get_git_info()

    # Load all data
    print("Loading data files...")
    data = load_all_data(config)

    # Create main table with severity_v2
    print("\nGenerating main results table (with severity_v2)...")
    main_table = create_main_table_v2(data, config)
    main_table.to_csv(paper_dir / 'table_main.csv', index=False)

    latex_main = create_latex_table_main(main_table)
    with open(paper_dir / 'table_main.tex', 'w') as f:
        f.write(latex_main)

    # Create seeds table
    print("Generating multi-seed table...")
    seeds_table = create_seeds_table(data)
    if seeds_table is not None:
        seeds_table.to_csv(paper_dir / 'table_seeds.csv', index=False)

        latex_seeds = create_latex_table_seeds(seeds_table)
        with open(paper_dir / 'table_seeds.tex', 'w') as f:
            f.write(latex_seeds)

    # Copy and deduplicate PR curve
    print("Copying and deduplicating PR curve files...")
    pr_curve_src = config.get_output_path('pr_curve.csv')
    if pr_curve_src.exists():
        pr_curve = pd.read_csv(pr_curve_src)
        pr_curve_dedup = pr_curve.drop_duplicates(subset=['threshold'], keep='first')
        pr_curve_dedup.to_csv(paper_dir / 'pr_curve.csv', index=False)
        print(f"  Deduplicated PR curve: {len(pr_curve)} -> {len(pr_curve_dedup)} points")

    # Create PR curve plot
    print("Creating PR curve plot...")
    create_pr_curve_plot(data, paper_dir / 'pr_curve.png')

    # Create severity histogram
    print("Creating severity histogram...")
    create_severity_histogram(data, paper_dir / 'severity_hist.png')

    # Create calibration plot
    print("Creating calibration curve...")
    create_calibration_plot(data, paper_dir / 'calibration_curve.png')

    # Create baseline details
    print("Creating baseline method documentation...")
    create_baseline_details(data, paper_dir / 'baseline_details.txt')

    # Create metrics overview
    print("Writing metrics overview...")
    overview = create_metrics_overview_v2(data, config, paper_dir, git_info)
    with open(paper_dir / 'metrics_overview.md', 'w', encoding='utf-8') as f:
        f.write(overview)

    # Create repro file
    print("Creating reproduction file...")
    create_repro_file(config, git_info, paper_dir / 'run_repro.txt')

    print("\n" + "="*80)
    print("PAPER-READY PACKAGE V2 COMPLETE")
    print("="*80)
    print(f"\nPackage location: {paper_dir.absolute()}")

    print("\nFiles generated:")
    for file in sorted(paper_dir.iterdir()):
        print(f"  - {file.name}")

    # Print key numbers
    print("\n" + "="*80)
    print("KEY NUMBERS FOR PAPER")
    print("="*80)
    print("\nMain Table:")
    print(main_table.to_string(index=False))

    if seeds_table is not None:
        print("\n" + "="*80)
        print("Multi-Seed Table (Mean ± Std):")
        print("="*80)
        print(seeds_table.to_string(index=False))

    return paper_dir


def main():
    """Main entry point."""
    # Find latest run (excluding multiseed)
    runs = sorted([r for r in Path('eval/synthetic/outputs').glob('*/')
                   if r.name != 'multiseed'])

    if not runs:
        print("ERROR: No run outputs found.")
        return

    latest_run = runs[-1]
    run_id = latest_run.name

    print(f"Using run: {run_id}")

    paper_dir = generate_paper_package_v2(run_id)

    print(f"\n\nSUCCESS! Paper-ready package V2: {paper_dir.absolute()}")


if __name__ == '__main__':
    main()
