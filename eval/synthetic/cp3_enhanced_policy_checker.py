#!/usr/bin/env python3
"""
CP3 Enhanced: Run policy checker with severity scoring and PR-AUC (Checkpoint 1).

Extends the basic policy checker to compute continuous severity scores for
each case, enabling PR curve analysis and more nuanced evaluation beyond
binary classification.

Severity components:
- Lateness: How far past the delegation window the approval occurred
- Role penalty: Whether the right role approved (senior vs junior vs missing)

Outputs:
- predictions.csv (enhanced with severity scores)
- pr_curve.csv (precision-recall curve data)
- pr_curve.png (PR curve visualization)
- metrics.json (includes average_precision and brier_score)
"""

from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    brier_score_loss
)

from config import SyntheticConfig


def load_synthetic_event_log(csv_path: Path) -> pd.DataFrame:
    """Load synthetic event log CSV."""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def find_request_anchor(case_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Find timestamp of first request anchor event (Submit_Request)."""
    anchor_df = case_df[case_df['activity'] == 'Submit_Request']
    if len(anchor_df) > 0:
        return anchor_df.iloc[0]['timestamp']
    return None


def find_approvals(case_df: pd.DataFrame) -> List[dict]:
    """Find all approval events with role classification."""
    approvals = []
    approval_df = case_df[case_df['activity'].isin(['Senior_Approval', 'Junior_Approval'])]

    for idx, row in approval_df.iterrows():
        approvals.append({
            'ts': row['timestamp'],
            'resource': row.get('resource', 'UNKNOWN'),
            'role': row.get('role', 'unknown') if pd.notna(row.get('role')) else 'unknown',
            'activity': row['activity']
        })

    return approvals


def compute_severity_score(case_id: str, amount: float, request_ts: Optional[pd.Timestamp],
                           approvals: List[dict], T: float, delta_hours: float,
                           config: SyntheticConfig) -> Dict:
    """
    Compute policy severity score with continuous gradations.

    Severity formula:
    severity = w1 * lateness_component + w2 * role_component

    where:
    - lateness_component = min(lateness_hours / Delta, 1.0)
    - role_component = role_penalty based on approval type

    Returns dict with: severity, lateness_hours, role_category, outcome, is_violation
    """
    delta = timedelta(hours=delta_hours)
    requires_senior = (amount >= T)

    # Initialize defaults
    lateness_hours = 0.0
    role_category = 'not_applicable'
    role_penalty = 0.0
    outcome = 'not_applicable'
    is_violation = False

    if not requires_senior:
        # Policy doesn't apply
        severity = 0.0
    else:
        # Policy applies - compute severity
        if request_ts is None:
            # Missing request anchor - severe issue
            lateness_hours = float('inf')
            role_category = 'missing_approval'
            role_penalty = config.role_penalty_missing
            outcome = 'duty_unmet'
            is_violation = True
        else:
            # Find approvals within/outside delta window
            deadline = request_ts + delta
            approvals_within = [a for a in approvals if a['ts'] <= deadline]
            approvals_after = [a for a in approvals if a['ts'] > deadline]

            # Check for senior approval
            senior_within = next((a for a in approvals_within if a['role'] == 'senior'), None)
            senior_after = next((a for a in approvals_after if a['role'] == 'senior'), None)
            junior_within = next((a for a in approvals_within if a['role'] == 'junior'), None)
            junior_after = next((a for a in approvals_after if a['role'] == 'junior'), None)

            if senior_within:
                # Best case: senior approval within window
                lateness_hours = 0.0
                role_category = 'senior_approval'
                role_penalty = config.role_penalty_correct
                outcome = 'duty_met'
                is_violation = False
            elif junior_within:
                # Delegation: junior within window (allowed)
                lateness_hours = 0.0
                role_category = 'delegation'
                role_penalty = config.role_penalty_correct
                outcome = 'duty_met_via_delegation'
                is_violation = False
            elif senior_after:
                # Late senior approval
                lateness_hours = (senior_after['ts'] - deadline).total_seconds() / 3600
                role_category = 'senior_late'
                role_penalty = config.role_penalty_correct  # Right role, but late
                outcome = 'duty_unmet'
                is_violation = True
            elif junior_after:
                # Late junior approval
                lateness_hours = (junior_after['ts'] - deadline).total_seconds() / 3600
                role_category = 'delegation_late'
                role_penalty = config.role_penalty_wrong_role  # Wrong timing for delegation
                outcome = 'duty_unmet'
                is_violation = True
            else:
                # No approval at all
                lateness_hours = float('inf')
                role_category = 'missing_approval'
                role_penalty = config.role_penalty_missing
                outcome = 'duty_unmet'
                is_violation = True

        # Compute lateness component (normalized to [0, 1])
        if lateness_hours == float('inf'):
            lateness_component = 1.0
        else:
            lateness_component = min(lateness_hours / delta_hours, 1.0)

        # Compute total severity
        severity = (config.severity_weight_lateness * lateness_component +
                   config.severity_weight_role * role_penalty)

    return {
        'case_id': case_id,
        'amount': amount,
        'requires_senior': requires_senior,
        'outcome': outcome,
        'is_violation': is_violation,
        'severity': float(severity),
        'lateness_hours': float(lateness_hours) if lateness_hours != float('inf') else 999999.0,
        'role_category': role_category,
        'evidence': f'{role_category}, lateness={lateness_hours:.1f}h' if lateness_hours != float('inf') else f'{role_category}, no approval'
    }


def run_enhanced_policy_checker(df: pd.DataFrame, T: float, delta_hours: float,
                                config: SyntheticConfig) -> pd.DataFrame:
    """Run policy checker with severity scoring."""
    predictions = []

    for case_id in df['case_id'].unique():
        case_df = df[df['case_id'] == case_id].sort_values('timestamp')
        amount = case_df['amount'].iloc[0]

        request_ts = find_request_anchor(case_df)
        approvals = find_approvals(case_df)

        prediction = compute_severity_score(case_id, amount, request_ts, approvals,
                                            T, delta_hours, config)
        predictions.append(prediction)

    return pd.DataFrame(predictions)


def compute_pr_curve_metrics(ground_truth: pd.DataFrame, predictions: pd.DataFrame,
                             config: SyntheticConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute precision-recall curve and related metrics.

    Uses severity scores as continuous predictions for violation detection.
    """
    # Merge ground truth and predictions
    merged = ground_truth.merge(predictions, on='case_id', suffixes=('_gt', '_pred'))

    # Binary labels (ground truth)
    y_true = merged['is_violation_gt'].astype(int)

    # Severity scores as continuous predictions
    y_scores = merged['severity']

    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Average Precision (area under PR curve)
    avg_precision = average_precision_score(y_true, y_scores)

    # Brier score (calibration metric)
    # Convert severity to probability using sigmoid-like transformation
    y_prob = 1 / (1 + np.exp(-5 * (y_scores - 0.5)))  # Sigmoid centered at 0.5
    brier = brier_score_loss(y_true, y_prob)

    # Create PR curve dataframe
    pr_df = pd.DataFrame({
        'threshold': np.append(thresholds, 1.0),  # Add final threshold
        'precision': precision,
        'recall': recall
    })

    metrics = {
        'average_precision': float(avg_precision),
        'brier_score': float(brier),
        'num_violations': int(y_true.sum()),
        'num_total': int(len(y_true))
    }

    return pr_df, metrics


def create_pr_curve_plot(pr_df: pd.DataFrame, avg_precision: float,
                         config: SyntheticConfig):
    """Create precision-recall curve visualization."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot PR curve
    ax.plot(pr_df['recall'], pr_df['precision'], 'b-', linewidth=2,
            label=f'PR Curve (AP={avg_precision:.3f})')

    # Add baseline (random classifier)
    baseline = pr_df['precision'].iloc[-1]  # Prevalence
    ax.plot([0, 1], [baseline, baseline], 'r--', linewidth=1.5,
            label=f'Random Baseline ({baseline:.3f})')

    # Perfect classifier reference
    ax.plot([0, 1, 1], [1, 1, pr_df['precision'].iloc[-1]], 'g:', linewidth=1.5,
            label='Perfect Classifier', alpha=0.5)

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(f'Precision-Recall Curve (Average Precision = {avg_precision:.3f})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    output_path = config.get_output_path('pr_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved PR curve to: {output_path}")
    plt.close()


def print_statistics(predictions_df: pd.DataFrame, pr_metrics: Dict):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("ENHANCED POLICY CHECKER RESULTS (WITH SEVERITY SCORING)")
    print("="*80)

    total_cases = len(predictions_df)
    violations = predictions_df['is_violation'].sum()

    print(f"\nTotal cases: {total_cases}")
    print(f"Violations: {violations} ({100*violations/total_cases:.1f}%)")

    # Severity statistics
    print(f"\nSeverity Score Statistics:")
    print(f"  Mean:   {predictions_df['severity'].mean():.3f}")
    print(f"  Median: {predictions_df['severity'].median():.3f}")
    print(f"  Min:    {predictions_df['severity'].min():.3f}")
    print(f"  Max:    {predictions_df['severity'].max():.3f}")
    print(f"  Std:    {predictions_df['severity'].std():.3f}")

    # Role category distribution
    print(f"\nRole Category Distribution:")
    role_counts = predictions_df['role_category'].value_counts()
    for role, count in role_counts.items():
        print(f"  {role:25s}: {count:5d} ({100*count/total_cases:.1f}%)")

    # PR metrics
    print(f"\nPrecision-Recall Metrics:")
    print(f"  Average Precision (PR-AUC): {pr_metrics['average_precision']:.3f}")
    print(f"  Brier Score:                {pr_metrics['brier_score']:.3f}")


def main():
    """Main execution for CP3 Enhanced."""
    print("="*80)
    print("CP3 ENHANCED: POLICY CHECKER WITH SEVERITY SCORING")
    print("="*80)

    # Load configuration
    runs = sorted(Path('eval/synthetic/outputs').glob('*/'))
    if not runs:
        print("ERROR: No outputs found. Run CP1 and CP2 first.")
        return

    latest_run = runs[-1]
    run_id = latest_run.name

    config = SyntheticConfig(run_id=run_id)
    print(f"\nConfiguration:")
    print(f"  Run ID: {config.run_id}")
    print(f"  Threshold (T): ${config.senior_approval_threshold:,.2f}")
    print(f"  Delegation Window (Delta): {config.delegation_window_hours}h")
    print(f"  Severity Weights: lateness={config.severity_weight_lateness}, "
          f"role={config.severity_weight_role}")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    log_path = config.get_output_path('event_log.csv')
    gt_path = config.get_output_path('ground_truth.csv')

    df = load_synthetic_event_log(log_path)
    ground_truth = pd.read_csv(gt_path)

    print(f"Loaded {len(df)} events from {df['case_id'].nunique()} cases")
    print(f"Ground truth: {len(ground_truth)} cases")

    # Run enhanced policy checker
    print("\n" + "="*80)
    print("RUNNING ENHANCED POLICY CHECKER")
    print("="*80)
    predictions_df = run_enhanced_policy_checker(
        df, config.senior_approval_threshold,
        config.delegation_window_hours, config
    )

    # Compute PR curve metrics
    print("\n" + "="*80)
    print("COMPUTING PR CURVE METRICS")
    print("="*80)
    pr_df, pr_metrics = compute_pr_curve_metrics(ground_truth, predictions_df, config)

    # Print statistics
    print_statistics(predictions_df, pr_metrics)

    # Save outputs
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)

    # Save enhanced predictions
    pred_path = config.get_output_path('predictions.csv')
    predictions_df.to_csv(pred_path, index=False)
    print(f"Saved predictions with severity to: {pred_path}")

    # Save PR curve data
    pr_curve_path = config.get_output_path('pr_curve.csv')
    pr_df.to_csv(pr_curve_path, index=False)
    print(f"Saved PR curve data to: {pr_curve_path}")

    # Update metrics.json with PR metrics
    metrics_path = config.get_output_path('metrics.json')
    if metrics_path.exists():
        import json
        with open(metrics_path, 'r') as f:
            existing_metrics = json.load(f)
        existing_metrics['pr_curve_metrics'] = pr_metrics
        with open(metrics_path, 'w') as f:
            json.dump(existing_metrics, f, indent=2)
        print(f"Updated metrics.json with PR metrics")
    else:
        import json
        with open(metrics_path, 'w') as f:
            json.dump({'pr_curve_metrics': pr_metrics}, f, indent=2)
        print(f"Created metrics.json with PR metrics")

    # Create PR curve plot
    print("\n" + "="*80)
    print("GENERATING PR CURVE PLOT")
    print("="*80)
    create_pr_curve_plot(pr_df, pr_metrics['average_precision'], config)

    print("\n" + "="*80)
    print("CP3 ENHANCED COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {pred_path}")
    print(f"  - {pr_curve_path}")
    print(f"  - {config.get_output_path('pr_curve.png')}")
    print(f"  - {metrics_path} (updated)")


if __name__ == '__main__':
    main()
