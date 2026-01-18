#!/usr/bin/env python3
"""
CP5: Parameter sweep over T (threshold) and Delta (delegation window).

Runs the policy checker with different parameter combinations to show:
- Sensitivity to threshold T
- Sensitivity to delegation window Delta
- Trade-offs between precision and recall

Outputs:
- sweep_results.csv (all combinations)
- sweep_heatmap_violations.png (violation rate heatmap)
- sweep_metrics_grid.png (precision/recall/F1 across parameters)
"""

from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

from config import SyntheticConfig


def load_data(config: SyntheticConfig) -> tuple:
    """Load event log and ground truth."""
    log_path = config.get_output_path('event_log.csv')
    gt_path = config.get_output_path('ground_truth.csv')

    df = pd.read_csv(log_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    ground_truth = pd.read_csv(gt_path)

    return df, ground_truth


def evaluate_with_parameters(df: pd.DataFrame, ground_truth: pd.DataFrame,
                             T: float, delta_hours: float) -> Dict:
    """
    Run policy checker with specific T and Delta parameters.

    Returns metrics for this parameter combination.
    """
    from cp3_run_policy_checker import find_request_anchor, find_approvals, evaluate_p1_duty

    predictions = []

    # Process each case
    for case_id in df['case_id'].unique():
        case_df = df[df['case_id'] == case_id].sort_values('timestamp')
        amount = case_df['amount'].iloc[0]

        # Find request anchor timestamp
        request_ts = find_request_anchor(case_df)

        # Find all approvals with role classification
        approvals = find_approvals(case_df)

        # Evaluate policy with current parameters
        prediction = evaluate_p1_duty(case_id, amount, request_ts, approvals, T, delta_hours)
        predictions.append(prediction)

    predictions_df = pd.DataFrame(predictions)

    # Merge with ground truth (using original parameters)
    merged = ground_truth.merge(predictions_df, on='case_id', suffixes=('_gt', '_pred'))

    # Compute metrics
    # Note: Ground truth is based on T=20000, Delta=24
    # We're comparing against predictions with varying T and Delta

    # Count outcomes
    total = len(predictions_df)
    not_applicable = (predictions_df['outcome'] == 'not_applicable').sum()
    duty_met = (predictions_df['outcome'] == 'duty_met').sum()
    duty_met_via_delegation = (predictions_df['outcome'] == 'duty_met_via_delegation').sum()
    duty_unmet = (predictions_df['outcome'] == 'duty_unmet').sum()

    applicable = total - not_applicable
    violation_rate = duty_unmet / applicable if applicable > 0 else 0

    # Binary classification metrics (vs ground truth)
    y_true = (merged['outcome_gt'] == 'duty_unmet').astype(int)
    y_pred = (merged['outcome_pred'] == 'duty_unmet').astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Agreement with ground truth
    agreement = (merged['outcome_gt'] == merged['outcome_pred']).sum() / len(merged)

    return {
        'T': T,
        'delta_hours': delta_hours,
        'total_cases': total,
        'not_applicable': int(not_applicable),
        'duty_met': int(duty_met),
        'duty_met_via_delegation': int(duty_met_via_delegation),
        'duty_unmet': int(duty_unmet),
        'applicable_cases': int(applicable),
        'violation_rate': float(violation_rate),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'agreement_with_gt': float(agreement)
    }


def run_parameter_sweep(df: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Run parameter sweep over T and Delta.

    Grid:
    - T: [15000, 20000, 25000, 30000] (original is 20000)
    - Delta: [12, 24, 48, 72] hours (original is 24)
    """
    T_values = [15000, 20000, 25000, 30000]
    delta_values = [12, 24, 48, 72]

    results = []

    total_combinations = len(T_values) * len(delta_values)
    print(f"Running {total_combinations} parameter combinations...")

    for i, T in enumerate(T_values, 1):
        for j, delta in enumerate(delta_values, 1):
            combo_num = (i - 1) * len(delta_values) + j
            print(f"  [{combo_num}/{total_combinations}] T=${T:,}, Delta={delta}h...", end=' ')

            result = evaluate_with_parameters(df, ground_truth, T, delta)
            results.append(result)

            print(f"violations={result['duty_unmet']}, F1={result['f1_score']:.3f}")

    return pd.DataFrame(results)


def create_violation_rate_heatmap(sweep_df: pd.DataFrame, config: SyntheticConfig):
    """Create heatmap of violation rates across T and Delta."""
    # Pivot for heatmap
    heatmap_data = sweep_df.pivot_table(
        values='violation_rate',
        index='delta_hours',
        columns='T',
        aggfunc='mean'
    )

    # Convert to percentage
    heatmap_data = heatmap_data * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Violation Rate (%)'},
                linewidths=1, linecolor='black',
                vmin=0, vmax=100, ax=ax)

    ax.set_xlabel('Threshold T (Amount)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Delegation Window Delta (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Sensitivity: Violation Rate vs T and Delta',
                 fontsize=14, fontweight='bold', pad=20)

    # Format labels
    ax.set_xticklabels([f'${int(x):,}' for x in heatmap_data.columns], rotation=0)
    ax.set_yticklabels([f'{int(y)}h' for y in heatmap_data.index], rotation=0)

    # Highlight original parameters (T=20000, Delta=24)
    # Find position in heatmap
    t_idx = list(heatmap_data.columns).index(20000)
    d_idx = list(heatmap_data.index).index(24)

    # Add rectangle around baseline cell
    from matplotlib.patches import Rectangle
    rect = Rectangle((t_idx, d_idx), 1, 1, fill=False, edgecolor='blue',
                     linewidth=3, linestyle='--')
    ax.add_patch(rect)

    # Add legend for baseline
    ax.text(0.02, 0.98, 'Blue box = baseline (T=$20K, Delta=24h)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    output_path = config.get_output_path('sweep_heatmap_violations.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved violation rate heatmap to: {output_path}")
    plt.close()


def create_metrics_grid(sweep_df: pd.DataFrame, config: SyntheticConfig):
    """Create grid of subplots showing precision/recall/F1 across parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get unique values
    T_values = sorted(sweep_df['T'].unique())
    delta_values = sorted(sweep_df['delta_hours'].unique())

    # 1. Violation rate heatmap (top-left)
    ax = axes[0, 0]
    heatmap_viol = sweep_df.pivot_table(values='violation_rate', index='delta_hours',
                                        columns='T', aggfunc='mean') * 100
    sns.heatmap(heatmap_viol, annot=True, fmt='.1f', cmap='OrRd',
                ax=ax, cbar_kws={'label': '%'})
    ax.set_title('Violation Rate (%)', fontweight='bold')
    ax.set_xlabel('T')
    ax.set_ylabel('Delta (h)')

    # 2. Precision heatmap (top-right)
    ax = axes[0, 1]
    heatmap_prec = sweep_df.pivot_table(values='precision', index='delta_hours',
                                        columns='T', aggfunc='mean')
    sns.heatmap(heatmap_prec, annot=True, fmt='.3f', cmap='Greens',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title('Precision', fontweight='bold')
    ax.set_xlabel('T')
    ax.set_ylabel('Delta (h)')

    # 3. Recall heatmap (bottom-left)
    ax = axes[1, 0]
    heatmap_rec = sweep_df.pivot_table(values='recall', index='delta_hours',
                                       columns='T', aggfunc='mean')
    sns.heatmap(heatmap_rec, annot=True, fmt='.3f', cmap='Blues',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title('Recall', fontweight='bold')
    ax.set_xlabel('T')
    ax.set_ylabel('Delta (h)')

    # 4. F1 heatmap (bottom-right)
    ax = axes[1, 1]
    heatmap_f1 = sweep_df.pivot_table(values='f1_score', index='delta_hours',
                                      columns='T', aggfunc='mean')
    sns.heatmap(heatmap_f1, annot=True, fmt='.3f', cmap='Purples',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title('F1-Score', fontweight='bold')
    ax.set_xlabel('T')
    ax.set_ylabel('Delta (h)')

    plt.suptitle('Parameter Sweep: Metrics Across T and Delta',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = config.get_output_path('sweep_metrics_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics grid to: {output_path}")
    plt.close()


def print_summary(sweep_df: pd.DataFrame):
    """Print summary statistics from sweep."""
    print("\n" + "="*80)
    print("PARAMETER SWEEP SUMMARY")
    print("="*80)

    # Baseline (T=20000, Delta=24)
    baseline = sweep_df[(sweep_df['T'] == 20000) & (sweep_df['delta_hours'] == 24)]
    if len(baseline) > 0:
        baseline = baseline.iloc[0]
        print(f"\nBaseline (T=$20,000, Delta=24h):")
        print(f"  Violation rate: {baseline['violation_rate']*100:.1f}%")
        print(f"  Precision: {baseline['precision']:.3f}")
        print(f"  Recall: {baseline['recall']:.3f}")
        print(f"  F1-Score: {baseline['f1_score']:.3f}")
        print(f"  Agreement with GT: {baseline['agreement_with_gt']*100:.1f}%")

    # Range of violation rates
    print(f"\nViolation rate range:")
    print(f"  Min: {sweep_df['violation_rate'].min()*100:.1f}% "
          f"(T=${sweep_df.loc[sweep_df['violation_rate'].idxmin(), 'T']:,.0f}, "
          f"Delta={sweep_df.loc[sweep_df['violation_rate'].idxmin(), 'delta_hours']:.0f}h)")
    print(f"  Max: {sweep_df['violation_rate'].max()*100:.1f}% "
          f"(T=${sweep_df.loc[sweep_df['violation_rate'].idxmax(), 'T']:,.0f}, "
          f"Delta={sweep_df.loc[sweep_df['violation_rate'].idxmax(), 'delta_hours']:.0f}h)")

    # Best F1 score
    best_f1_idx = sweep_df['f1_score'].idxmax()
    best_f1 = sweep_df.loc[best_f1_idx]
    print(f"\nBest F1-Score:")
    print(f"  F1: {best_f1['f1_score']:.3f}")
    print(f"  T=${best_f1['T']:,.0f}, Delta={best_f1['delta_hours']:.0f}h")
    print(f"  Precision: {best_f1['precision']:.3f}, Recall: {best_f1['recall']:.3f}")

    # Agreement with ground truth
    print(f"\nAgreement with ground truth:")
    print(f"  Min: {sweep_df['agreement_with_gt'].min()*100:.1f}%")
    print(f"  Max: {sweep_df['agreement_with_gt'].max()*100:.1f}%")


def main():
    """Main execution for CP5."""
    print("="*80)
    print("CP5: PARAMETER SWEEP")
    print("="*80)

    # Load configuration
    runs = sorted(Path('eval/synthetic/outputs').glob('*/'))
    if not runs:
        print("ERROR: No outputs found. Run CP1-CP4 first.")
        return

    latest_run = runs[-1]
    run_id = latest_run.name

    config = SyntheticConfig(run_id=run_id)
    print(f"\nRun ID: {config.run_id}")
    print(f"Baseline parameters: T=${config.senior_approval_threshold:,.2f}, "
          f"Delta={config.delegation_window_hours}h")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    df, ground_truth = load_data(config)
    print(f"Loaded {len(df)} events from {df['case_id'].nunique()} cases")
    print(f"Ground truth: {len(ground_truth)} cases")

    # Run parameter sweep
    print("\n" + "="*80)
    print("RUNNING PARAMETER SWEEP")
    print("="*80)
    sweep_df = run_parameter_sweep(df, ground_truth)

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    output_path = config.get_output_path('sweep_results.csv')
    sweep_df.to_csv(output_path, index=False)
    print(f"Saved sweep results to: {output_path}")

    # Print summary
    print_summary(sweep_df)

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    create_violation_rate_heatmap(sweep_df, config)
    create_metrics_grid(sweep_df, config)

    print("\n" + "="*80)
    print("CP5 COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {output_path}")
    print(f"  - {config.get_output_path('sweep_heatmap_violations.png')}")
    print(f"  - {config.get_output_path('sweep_metrics_grid.png')}")


if __name__ == '__main__':
    main()
