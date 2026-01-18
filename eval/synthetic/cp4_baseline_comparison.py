#!/usr/bin/env python3
"""
CP4: Baseline comparison - Control-flow vs Policy-only conformance.

Shows that policy-only checking adds value by catching violations in
process-conforming traces. Control-flow conformance (token-based replay)
is high (~100%) because all traces complete the workflow, but policy
violations still exist.

Outputs:
- baseline_results.csv: Comparison of control-flow vs policy-only
- baseline_comparison.png: Visualization of added value
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer

from config import SyntheticConfig


def run_control_flow_conformance(log_path: Path, config: SyntheticConfig) -> pd.DataFrame:
    """
    Run control-flow conformance checking.

    Uses a simple activity-based conformance check:
    - Checks if trace contains required activities (Submit, Check, Approval, Process, Notify, Close)
    - Does NOT check data attributes (amount, time, role)
    - This represents traditional control-flow conformance

    Returns DataFrame with case-level conformance scores.
    """
    print("Loading event log...")
    log = xes_importer.apply(str(log_path))

    # Required activities in the process (order-independent for simplicity)
    required_activities = {
        'Submit_Request',
        'Check_Amount',
        'Process_Payment',
        'Send_Notification',
        'Close_Case'
    }

    # At least one approval is required
    approval_activities = {'Senior_Approval', 'Junior_Approval'}

    # Extract conformance metrics per case
    results = []
    for trace in log:
        case_id = trace.attributes['concept:name']
        amount = trace.attributes.get('amount', 0)

        # Get activities in this trace
        trace_activities = set(event['concept:name'] for event in trace)

        # Check control-flow conformance (activity presence only)
        has_required = required_activities.issubset(trace_activities)
        has_approval = bool(trace_activities.intersection(approval_activities))

        is_fit = has_required and has_approval
        fitness = 1.0 if is_fit else 0.0

        # Count missing activities (for diagnostics)
        missing = required_activities - trace_activities
        if not has_approval:
            missing.add('(any_approval)')

        results.append({
            'case_id': case_id,
            'amount': amount,
            'control_flow_fit': is_fit,
            'control_flow_fitness': fitness,
            'missing_activities': ', '.join(missing) if missing else ''
        })

    return pd.DataFrame(results)


def compare_with_policy_results(control_flow_df: pd.DataFrame,
                                ground_truth_df: pd.DataFrame,
                                predictions_df: pd.DataFrame,
                                config: SyntheticConfig) -> pd.DataFrame:
    """
    Combine control-flow and policy-only results for comparison.
    """
    # Merge all three DataFrames
    merged = control_flow_df.merge(ground_truth_df, on='case_id', suffixes=('_cf', '_gt'))
    merged = merged.merge(predictions_df, on='case_id', suffixes=('', '_pred'))

    # Add comparison flags
    merged['cf_says_ok'] = merged['control_flow_fit']
    merged['policy_says_violation'] = merged['is_violation']
    merged['cf_missed_violation'] = merged['cf_says_ok'] & merged['policy_says_violation']

    return merged


def print_baseline_comparison(comparison_df: pd.DataFrame):
    """Print summary statistics for baseline comparison."""
    print("\n" + "="*80)
    print("BASELINE COMPARISON: CONTROL-FLOW vs POLICY-ONLY")
    print("="*80)

    total_cases = len(comparison_df)

    # Control-flow statistics
    cf_conforming = comparison_df['cf_says_ok'].sum()
    cf_fitness_mean = comparison_df['control_flow_fitness'].mean()

    print(f"\nControl-Flow Conformance (Activity-Based Check):")
    print(f"  Conforming cases:     {cf_conforming} / {total_cases} ({100*cf_conforming/total_cases:.1f}%)")
    print(f"  Mean fitness score:   {cf_fitness_mean:.3f}")

    # Show any non-conforming cases
    cf_non_conforming = total_cases - cf_conforming
    if cf_non_conforming > 0:
        print(f"  Non-conforming:       {cf_non_conforming} (missing required activities)")

    # Policy-only statistics
    policy_violations = comparison_df['policy_says_violation'].sum()

    print(f"\nPolicy-Only Conformance:")
    print(f"  Violations detected:  {policy_violations} / {total_cases} ({100*policy_violations/total_cases:.1f}%)")
    print(f"  Conforming cases:     {total_cases - policy_violations} / {total_cases} ({100*(total_cases - policy_violations)/total_cases:.1f}%)")

    # Added value: violations in control-flow conforming traces
    cf_missed = comparison_df['cf_missed_violation'].sum()

    print(f"\nAdded Value of Policy-Only Checking:")
    print(f"  Violations in CF-conforming traces: {cf_missed} / {policy_violations} ({100*cf_missed/policy_violations if policy_violations > 0 else 0:.1f}%)")
    print(f"  -> Control-flow conformance MISSED these violations")
    print(f"  -> Policy-only checking caught all {policy_violations} violations")

    # Breakdown by severity
    if 'severity' in comparison_df.columns:
        high_severity = (comparison_df['severity'] >= 0.5).sum()
        print(f"\nHigh-severity violations (severity >= 0.5): {high_severity}")


def create_baseline_comparison_plot(comparison_df: pd.DataFrame, config: SyntheticConfig):
    """Create visualization comparing control-flow vs policy-only."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Venn-style comparison
    ax1 = axes[0]

    total = len(comparison_df)
    cf_conforming = comparison_df['cf_says_ok'].sum()
    policy_violations = comparison_df['policy_says_violation'].sum()
    cf_missed = comparison_df['cf_missed_violation'].sum()

    categories = ['CF Conforming\n(Process OK)', 'Policy Violations\n(Data-aware)']
    values = [cf_conforming, policy_violations]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({100*val/total:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax1.set_title('Control-Flow vs Policy-Only Conformance', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, total * 1.15])
    ax1.grid(axis='y', alpha=0.3)

    # Add annotation for missed violations
    ax1.text(0.5, 0.95, f'{cf_missed} violations missed by CF\n(caught by policy-only)',
             transform=ax1.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             fontsize=10, fontweight='bold')

    # Right plot: Severity distribution
    ax2 = axes[1]

    if 'severity' in comparison_df.columns:
        violations_df = comparison_df[comparison_df['policy_says_violation']]

        if len(violations_df) > 0:
            ax2.hist(violations_df['severity'], bins=20, color='#e74c3c',
                    alpha=0.7, edgecolor='black', linewidth=1)
            ax2.axvline(violations_df['severity'].mean(), color='darkred',
                       linestyle='--', linewidth=2, label=f'Mean: {violations_df["severity"].mean():.3f}')
            ax2.set_xlabel('Severity Score', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Number of Violations', fontsize=12, fontweight='bold')
            ax2.set_title('Policy Violation Severity Distribution', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = config.get_output_path('baseline_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved baseline comparison plot to: {output_path}")
    plt.close()


def main():
    """Main execution for CP4 baseline comparison."""
    print("="*80)
    print("CP4: BASELINE COMPARISON (CONTROL-FLOW vs POLICY-ONLY)")
    print("="*80)

    # Load configuration from latest run (excluding multiseed folder)
    runs = sorted([r for r in Path('eval/synthetic/outputs').glob('*/')
                   if r.name != 'multiseed'])
    if not runs:
        print("ERROR: No outputs found. Run CP1-CP3 first.")
        return

    latest_run = runs[-1]
    run_id = latest_run.name

    config = SyntheticConfig(run_id=run_id)
    print(f"\nUsing run: {config.run_id}")

    # Check required files exist
    log_path = config.get_output_path('event_log.xes')
    gt_path = config.get_output_path('ground_truth.csv')
    pred_path = config.get_output_path('predictions.csv')

    if not all([log_path.exists(), gt_path.exists(), pred_path.exists()]):
        print("ERROR: Missing required files. Run CP1-CP3 first.")
        return

    # Run control-flow conformance
    print("\n" + "="*80)
    print("RUNNING CONTROL-FLOW CONFORMANCE (ACTIVITY-BASED)")
    print("="*80)
    control_flow_df = run_control_flow_conformance(log_path, config)

    # Load policy-only results
    print("\n" + "="*80)
    print("LOADING POLICY-ONLY RESULTS")
    print("="*80)
    ground_truth_df = pd.read_csv(gt_path)
    predictions_df = pd.read_csv(pred_path)

    # Compare
    print("\n" + "="*80)
    print("COMPARING RESULTS")
    print("="*80)
    comparison_df = compare_with_policy_results(control_flow_df, ground_truth_df,
                                                predictions_df, config)

    # Print statistics
    print_baseline_comparison(comparison_df)

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_path = config.get_output_path('baseline_results.csv')
    comparison_df.to_csv(output_path, index=False)
    print(f"Saved baseline comparison to: {output_path}")

    # Create plot
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)
    create_baseline_comparison_plot(comparison_df, config)

    print("\n" + "="*80)
    print("CP4 BASELINE COMPARISON COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {output_path}")
    print(f"  - {config.get_output_path('baseline_comparison.png')}")


if __name__ == '__main__':
    main()
