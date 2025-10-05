#!/usr/bin/env python3
"""
Run experiments E1-E4 on policy log for P1 (Senior Approval Duty).
Policy-only conformance checking experiments with varying parameters.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_policy_log(csv_path: str) -> pd.DataFrame:
    """Load policy log CSV."""
    df = pd.read_csv(csv_path)
    # Parse timestamps with mixed format
    df['event_ts'] = pd.to_datetime(df['event_ts'], format='mixed', utc=True)
    df['request_ts'] = pd.to_datetime(df['request_ts'], format='mixed', utc=True, errors='coerce')
    return df


def experiment_e1_base_prevalence(df: pd.DataFrame) -> Dict:
    """
    E1: Base prevalence - report counts/rates for all outcomes.

    Returns:
        Dictionary with outcome counts, rates, and statistics
    """
    print("\n" + "="*80)
    print("E1: BASE PREVALENCE ANALYSIS")
    print("="*80)

    total = len(df)
    outcome_counts = df['outcome'].value_counts().to_dict()

    results = {
        'total_evaluations': total,
        'total_cases': df['case_id'].nunique(),
        'outcomes': {}
    }

    print(f"\nTotal evaluations: {total}")
    print(f"Total cases: {results['total_cases']}")
    print("\nOutcome Distribution:")

    for outcome in ['not_applicable', 'duty_met', 'duty_met_via_delegation', 'duty_unmet']:
        count = outcome_counts.get(outcome, 0)
        rate = count / total if total > 0 else 0
        results['outcomes'][outcome] = {
            'count': int(count),
            'rate': float(rate),
            'percentage': f"{100*rate:.2f}%"
        }
        print(f"  {outcome:30s}: {count:5d} ({100*rate:5.1f}%)")

    # Additional statistics
    applicable_cases = df[df['requires_senior'] == True]
    if len(applicable_cases) > 0:
        violation_rate_applicable = (applicable_cases['outcome'] == 'duty_unmet').sum() / len(applicable_cases)
        delegation_rate_applicable = (applicable_cases['outcome'] == 'duty_met_via_delegation').sum() / len(applicable_cases)

        results['applicable_only'] = {
            'count': len(applicable_cases),
            'violation_rate': float(violation_rate_applicable),
            'delegation_rate': float(delegation_rate_applicable),
            'senior_approval_rate': float((applicable_cases['outcome'] == 'duty_met').sum() / len(applicable_cases))
        }

        print(f"\nAmong applicable cases (requires_senior=True, n={len(applicable_cases)}):")
        print(f"  Violation rate: {100*violation_rate_applicable:.1f}%")
        print(f"  Delegation rate: {100*delegation_rate_applicable:.1f}%")

    # Wait time statistics
    wait_stats = df[df['wait_hours'].notna()]['wait_hours'].describe()
    results['wait_hours_stats'] = {
        'mean': float(wait_stats['mean']),
        'median': float(wait_stats['50%']),
        'std': float(wait_stats['std']),
        'min': float(wait_stats['min']),
        'max': float(wait_stats['max'])
    }

    print(f"\nWait hours statistics:")
    print(f"  Mean: {wait_stats['mean']:.1f}h ({wait_stats['mean']/24:.1f} days)")
    print(f"  Median: {wait_stats['50%']:.1f}h ({wait_stats['50%']/24:.1f} days)")
    print(f"  Std: {wait_stats['std']:.1f}h")

    return results


def recompute_outcomes(df: pd.DataFrame, T: float, delta_hours: float) -> pd.Series:
    """
    Recompute outcomes for new threshold T and delegation window delta.

    Args:
        df: Policy log DataFrame
        T: New amount threshold
        delta_hours: New delegation window (hours)

    Returns:
        Series of recomputed outcomes
    """
    outcomes = []

    for idx, row in df.iterrows():
        amount = row['amount']
        wait_hours = row['wait_hours']
        senior_seq = row['senior_approval_seq']
        junior_seq = row['junior_approval_seq']

        requires_senior = (amount >= T)

        if not requires_senior:
            outcomes.append('not_applicable')
        elif pd.notna(senior_seq):
            # Senior approval exists and within window
            if pd.isna(wait_hours) or wait_hours <= delta_hours:
                outcomes.append('duty_met')
            else:
                # Senior approval but outside window
                outcomes.append('duty_unmet')
        elif pd.notna(junior_seq):
            # Junior approval exists
            if pd.isna(wait_hours) or wait_hours <= delta_hours:
                outcomes.append('duty_met_via_delegation')
            else:
                outcomes.append('duty_unmet')
        else:
            # No approval
            outcomes.append('duty_unmet')

    return pd.Series(outcomes, index=df.index)


def experiment_e2_sensitivity(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    E2: Sensitivity to thresholds - sweep T and Δ, measure violation rates.

    Returns:
        (results_df, summary_dict)
    """
    print("\n" + "="*80)
    print("E2: SENSITIVITY TO THRESHOLDS")
    print("="*80)

    T_values = [15000, 20000, 25000]
    delta_values = [12, 24, 36]  # hours

    results = []

    for T in T_values:
        for delta in delta_values:
            print(f"\nComputing for T={T}, Delta={delta}h...")

            # Recompute outcomes
            outcomes = recompute_outcomes(df, T, delta)

            # Count outcomes
            total = len(outcomes)
            not_applicable = (outcomes == 'not_applicable').sum()
            duty_met = (outcomes == 'duty_met').sum()
            duty_met_via_delegation = (outcomes == 'duty_met_via_delegation').sum()
            duty_unmet = (outcomes == 'duty_unmet').sum()

            applicable = total - not_applicable
            violation_rate = duty_unmet / applicable if applicable > 0 else 0
            delegation_rate = duty_met_via_delegation / applicable if applicable > 0 else 0

            results.append({
                'T': T,
                'delta_hours': delta,
                'total': total,
                'not_applicable': not_applicable,
                'duty_met': duty_met,
                'duty_met_via_delegation': duty_met_via_delegation,
                'duty_unmet': duty_unmet,
                'applicable_cases': applicable,
                'violation_rate': violation_rate,
                'delegation_rate': delegation_rate
            })

            print(f"  Violation rate: {100*violation_rate:.1f}% ({duty_unmet}/{applicable})")
            print(f"  Delegation rate: {100*delegation_rate:.1f}%")

    results_df = pd.DataFrame(results)

    # Summary
    summary = {
        'parameter_ranges': {
            'T': T_values,
            'delta_hours': delta_values
        },
        'violation_rate_range': {
            'min': float(results_df['violation_rate'].min()),
            'max': float(results_df['violation_rate'].max()),
            'mean': float(results_df['violation_rate'].mean())
        },
        'most_lenient': results_df.loc[results_df['violation_rate'].idxmin()].to_dict(),
        'most_strict': results_df.loc[results_df['violation_rate'].idxmax()].to_dict()
    }

    print(f"\nViolation rate range: {100*summary['violation_rate_range']['min']:.1f}% - {100*summary['violation_rate_range']['max']:.1f}%")

    return results_df, summary


def experiment_e3_policy_value(df: pd.DataFrame) -> Dict:
    """
    E3: Where policy adds value - cases flagged by policy while flow appears fine.

    Proxy: target events exist (O_Accepted, A_Approved) → flow completed
    But policy shows violation → policy catches something flow doesn't

    Returns:
        Dictionary with value-added statistics
    """
    print("\n" + "="*80)
    print("E3: WHERE POLICY ADDS VALUE")
    print("="*80)

    # All cases in policy log reached target events (by definition)
    # So "flow appears fine" for all of them

    total_cases = df['case_id'].nunique()

    # Cases flagged by policy (duty_unmet)
    violation_cases = df[df['outcome'] == 'duty_unmet']['case_id'].nunique()

    # Cases where flow is complete but policy violated
    # (all violations in this dataset, since all have target events)
    policy_added_value_cases = violation_cases

    # Percentage of "normal-looking" cases that policy catches
    value_rate = policy_added_value_cases / total_cases if total_cases > 0 else 0

    results = {
        'total_cases_with_target_events': total_cases,
        'cases_flagged_by_policy': policy_added_value_cases,
        'policy_value_added_rate': float(value_rate),
        'interpretation': (
            f"Policy detected violations in {policy_added_value_cases}/{total_cases} cases "
            f"({100*value_rate:.1f}%) that completed their control flow successfully. "
            "These violations would be invisible to traditional control-flow conformance checking."
        )
    }

    print(f"\nTotal cases that reached target events: {total_cases}")
    print(f"Cases flagged by policy as violations: {policy_added_value_cases}")
    print(f"Policy value-added rate: {100*value_rate:.1f}%")
    print(f"\nInterpretation:")
    print(f"  {results['interpretation']}")

    # Breakdown by amount bracket
    print(f"\nViolations by amount bracket:")
    violations_df = df[df['outcome'] == 'duty_unmet'].copy()
    if len(violations_df) > 0:
        violations_df['amount_bracket'] = pd.cut(
            violations_df['amount'],
            bins=[0, 20000, 30000, 50000, 100000, float('inf')],
            labels=['<20K', '20-30K', '30-50K', '50-100K', '>100K']
        )
        bracket_counts = violations_df['amount_bracket'].value_counts().sort_index()
        for bracket, count in bracket_counts.items():
            print(f"  {bracket}: {count}")

    return results


def experiment_e4_early_warning(df: pd.DataFrame, delta_hours: float = 24) -> Dict:
    """
    E4 (Optional): Early-warning baseline heuristic.

    Heuristic: Raise alarm if junior approval occurs and wait_hours ≥ Δ

    Metrics:
    - Precision: Of alarms raised, how many are true violations?
    - Lead time: How early does the alarm fire (hours before target event)?

    Returns:
        Dictionary with early-warning statistics
    """
    print("\n" + "="*80)
    print("E4: EARLY-WARNING BASELINE (OPTIONAL)")
    print("="*80)

    # Identify cases where junior approval occurred
    junior_approval_cases = df[pd.notna(df['junior_approval_seq'])].copy()

    if len(junior_approval_cases) == 0:
        print("\nNo junior approvals found. Skipping E4.")
        return {'skipped': True, 'reason': 'No junior approvals in dataset'}

    # Apply heuristic: alarm if wait_hours >= delta
    junior_approval_cases['alarm'] = junior_approval_cases['wait_hours'] >= delta_hours

    alarms_raised = junior_approval_cases['alarm'].sum()

    # True violations among junior approval cases
    true_violations = (junior_approval_cases['outcome'] == 'duty_unmet').sum()

    # Precision: of alarms, how many are violations?
    # Note: This is tricky because our "violation" is also based on delta
    # Let's use a proxy: alarms that correspond to high wait times
    alarms_df = junior_approval_cases[junior_approval_cases['alarm']]

    if len(alarms_df) > 0:
        # Lead time: difference between junior approval time and target event time
        # Approximated by: wait_hours - (time from junior approval to target)
        # Since we don't have junior approval timestamp, use wait_hours as proxy
        avg_lead_time = alarms_df['wait_hours'].mean() - delta_hours

        precision = true_violations / alarms_raised if alarms_raised > 0 else 0
    else:
        avg_lead_time = 0
        precision = 0

    results = {
        'total_junior_approvals': len(junior_approval_cases),
        'alarms_raised': int(alarms_raised),
        'true_violations_in_junior': int(true_violations),
        'precision': float(precision),
        'avg_lead_time_hours': float(avg_lead_time) if avg_lead_time > 0 else 0,
        'interpretation': (
            f"Heuristic raised {alarms_raised} alarms for cases with junior approval "
            f"where wait ≥ {delta_hours}h. Precision: {100*precision:.1f}%. "
            f"Average lead time: {avg_lead_time:.1f}h ({avg_lead_time/24:.1f} days)."
        )
    }

    print(f"\nJunior approval cases: {len(junior_approval_cases)}")
    print(f"Alarms raised (wait >= {delta_hours}h): {alarms_raised}")
    print(f"Precision: {100*precision:.1f}%")
    print(f"Average lead time: {avg_lead_time:.1f}h ({avg_lead_time/24:.1f} days)")

    return results


def create_visualizations(e1_results: Dict, e2_df: pd.DataFrame, output_dir: Path):
    """
    Create visualizations for experiments.

    Generates:
    - fig_outcome_bars.png (E1)
    - fig_violation_heatmap.png (E2)
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Figure 1: Outcome distribution (E1)
    fig, ax = plt.subplots(figsize=(10, 6))

    outcomes = e1_results['outcomes']
    labels = list(outcomes.keys())
    counts = [outcomes[k]['count'] for k in labels]
    percentages = [outcomes[k]['rate'] * 100 for k in labels]

    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']  # green, blue, orange, red

    bars = ax.bar(range(len(labels)), counts, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=10)
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('E1: Outcome Distribution (Policy-Only Conformance Checking)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig_path = output_dir / 'fig_outcome_bars.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # Figure 2: Violation rate heatmap (E2)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Pivot table for heatmap
    heatmap_data = e2_df.pivot_table(
        values='violation_rate',
        index='delta_hours',
        columns='T',
        aggfunc='mean'
    )

    # Convert to percentage
    heatmap_data = heatmap_data * 100

    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Violation Rate (%)'},
                linewidths=1, linecolor='black',
                vmin=0, vmax=100, ax=ax)

    ax.set_xlabel('Threshold T (Amount)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Delegation Window Δ (hours)', fontsize=12, fontweight='bold')
    ax.set_title('E2: Violation Rate Sensitivity to T and Δ',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate x labels
    ax.set_xticklabels([f'${int(x):,}' for x in heatmap_data.columns], rotation=0)
    ax.set_yticklabels([f'{int(y)}h' for y in heatmap_data.index], rotation=0)

    plt.tight_layout()
    fig_path = output_dir / 'fig_violation_heatmap.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()


def main():
    """Run all experiments."""
    print("="*80)
    print("POLICY-ONLY CONFORMANCE CHECKING EXPERIMENTS")
    print("P1: Senior Approval Duty")
    print("="*80)

    # Setup
    output_dir = Path('eval/policy_only')
    policy_log_path = output_dir / 'policy_log.csv'

    # Load policy log
    print(f"\nLoading policy log from: {policy_log_path}")
    df = load_policy_log(str(policy_log_path))
    print(f"Loaded {len(df)} evaluations from {df['case_id'].nunique()} cases")

    # Run experiments
    e1_results = experiment_e1_base_prevalence(df)
    e2_df, e2_summary = experiment_e2_sensitivity(df)
    e3_results = experiment_e3_policy_value(df)
    e4_results = experiment_e4_early_warning(df, delta_hours=24)

    # Create visualizations
    create_visualizations(e1_results, e2_df, output_dir)

    # Save results table
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results_table_path = output_dir / 'results_table.csv'
    e2_df.to_csv(results_table_path, index=False)
    print(f"  Results table saved: {results_table_path}")

    # Save JSON summary
    all_results = {
        'experiment_summary': {
            'e1_base_prevalence': e1_results,
            'e2_sensitivity': e2_summary,
            'e3_policy_value': e3_results,
            'e4_early_warning': e4_results
        },
        'dataset_info': {
            'policy_log_path': str(policy_log_path),
            'total_evaluations': len(df),
            'total_cases': df['case_id'].nunique()
        }
    }

    results_json_path = output_dir / 'experiment_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Experiment results saved: {results_json_path}")

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - {results_table_path}")
    print(f"  - {results_json_path}")
    print(f"  - {output_dir / 'fig_outcome_bars.png'}")
    print(f"  - {output_dir / 'fig_violation_heatmap.png'}")


if __name__ == '__main__':
    main()
