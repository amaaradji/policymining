#!/usr/bin/env python3
"""
CP3: Run policy-only checker on synthetic log to generate predictions.

Reuses the policy evaluation logic from eval/policy_only/generate_policy_log_p1.py
but adapted for the synthetic log format.

Outputs:
- predictions.csv (case-level predictions from policy checker)
"""

from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import SyntheticConfig


def load_synthetic_event_log(csv_path: Path) -> pd.DataFrame:
    """Load synthetic event log CSV."""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def find_request_anchor(case_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """
    Find timestamp of first request anchor event in case.
    For synthetic log: Submit_Request
    """
    anchor_df = case_df[case_df['activity'] == 'Submit_Request']
    if len(anchor_df) > 0:
        return anchor_df.iloc[0]['timestamp']
    return None


def find_approvals(case_df: pd.DataFrame) -> List[dict]:
    """
    Find all approval events with role classification.

    For synthetic log:
    - Senior_Approval events have role='senior'
    - Junior_Approval events have role='junior'
    """
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


def evaluate_p1_duty(case_id: str, amount: float, request_ts: Optional[pd.Timestamp],
                     approvals: List[dict], T: float, delta_hours: float) -> dict:
    """
    Evaluate P1 duty (senior approval requirement) for a single case.

    Policy Logic (same as BPIC checker):
    - If amount >= T: senior approval is required (duty)
    - Senior approval within delta_hours: duty_met
    - Junior approval within delta_hours (no senior): duty_met_via_delegation
    - No approval within delta_hours: duty_unmet (violation)

    Args:
        case_id: Case identifier
        amount: Purchase amount for this case
        request_ts: Timestamp of request anchor (Submit_Request)
        approvals: List of approval events with role classification
        T: Amount threshold
        delta_hours: Delegation window in hours

    Returns:
        Dictionary with prediction for this case
    """
    delta = timedelta(hours=delta_hours)

    requires_senior = (amount >= T)

    # Filter approvals within delta window from request anchor
    approvals_within_delta = []
    if request_ts is not None:
        approvals_within_delta = [a for a in approvals if a['ts'] <= request_ts + delta]
    else:
        approvals_within_delta = approvals

    # Find senior approval within delta window
    senior_approval = next((a for a in approvals_within_delta if a.get('role') == 'senior'), None)

    # Find junior approval within delta
    junior_approval_within = next((a for a in approvals_within_delta if a.get('role') == 'junior'), None)

    # Determine outcome based on policy logic
    if not requires_senior:
        outcome = 'not_applicable'
        evidence = f'amount={amount:.2f} < T={T}'
    elif senior_approval is not None:
        outcome = 'duty_met'
        evidence = f'Senior approval within {delta_hours}h window'
    elif junior_approval_within is not None:
        outcome = 'duty_met_via_delegation'
        evidence = f'Junior approval (delegation) within {delta_hours}h window'
    else:
        outcome = 'duty_unmet'
        if request_ts is not None:
            evidence = f'VIOLATION: No approval within {delta_hours}h (amount={amount:.2f} >= T={T})'
        else:
            evidence = f'VIOLATION: No request anchor found, no approval detected'

    return {
        'case_id': case_id,
        'amount': amount,
        'requires_senior': requires_senior,
        'outcome': outcome,
        'is_violation': outcome == 'duty_unmet',
        'evidence': evidence
    }


def run_policy_checker(df: pd.DataFrame, T: float, delta_hours: float) -> pd.DataFrame:
    """
    Run policy-only checker on synthetic event log.

    Args:
        df: Event log DataFrame
        T: Amount threshold for senior approval
        delta_hours: Delegation window (hours)

    Returns:
        Predictions DataFrame (one row per case)
    """
    predictions = []

    # Process each case
    for case_id in df['case_id'].unique():
        case_df = df[df['case_id'] == case_id].sort_values('timestamp')
        amount = case_df['amount'].iloc[0]  # Amount is same for all events in a case

        # Find request anchor timestamp
        request_ts = find_request_anchor(case_df)

        # Find all approvals with role classification
        approvals = find_approvals(case_df)

        # Evaluate policy
        prediction = evaluate_p1_duty(case_id, amount, request_ts, approvals, T, delta_hours)
        predictions.append(prediction)

    predictions_df = pd.DataFrame(predictions)
    return predictions_df


def print_statistics(predictions_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("POLICY CHECKER PREDICTIONS")
    print("="*80)

    total_cases = len(predictions_df)
    print(f"Total cases evaluated: {total_cases}")

    # Outcome distribution
    print("\nPredicted outcome distribution:")
    outcome_counts = predictions_df['outcome'].value_counts()
    for outcome in ['not_applicable', 'duty_met', 'duty_met_via_delegation', 'duty_unmet']:
        count = outcome_counts.get(outcome, 0)
        pct = 100 * count / total_cases
        print(f"  {outcome:30s}: {count:5d} ({pct:5.1f}%)")

    # Violation statistics
    violations = predictions_df['is_violation'].sum()
    applicable_cases = predictions_df[predictions_df['requires_senior']].shape[0]
    violation_rate = violations / applicable_cases if applicable_cases > 0 else 0

    print(f"\nViolation statistics:")
    print(f"  Total violations predicted: {violations}")
    print(f"  Applicable cases: {applicable_cases}")
    print(f"  Violation rate (among applicable): {100*violation_rate:.1f}%")


def main():
    """Main execution for CP3."""
    print("="*80)
    print("CP3: RUN POLICY-ONLY CHECKER")
    print("="*80)

    # Load configuration (use latest run)
    runs = sorted(Path('eval/synthetic/outputs').glob('*/'))
    if not runs:
        print("ERROR: No CP2 output found. Run CP1 and CP2 first.")
        return

    latest_run = runs[-1]
    run_id = latest_run.name

    config = SyntheticConfig(run_id=run_id)
    print(f"\nConfiguration:")
    print(f"  Run ID: {config.run_id}")
    print(f"  Senior approval threshold (T): ${config.senior_approval_threshold:,.2f}")
    print(f"  Delegation window (Delta): {config.delegation_window_hours} hours")

    # Load synthetic event log
    print("\n" + "="*80)
    print("LOADING SYNTHETIC EVENT LOG")
    print("="*80)
    csv_path = config.get_output_path('event_log.csv')
    print(f"Loading from: {csv_path}")
    df = load_synthetic_event_log(csv_path)
    print(f"Loaded {len(df)} events from {df['case_id'].nunique()} cases")

    # Run policy checker
    print("\n" + "="*80)
    print("RUNNING POLICY CHECKER")
    print("="*80)
    predictions_df = run_policy_checker(
        df,
        T=config.senior_approval_threshold,
        delta_hours=config.delegation_window_hours
    )

    # Print statistics
    print_statistics(predictions_df)

    # Save predictions
    print("\n" + "="*80)
    print("SAVING PREDICTIONS")
    print("="*80)
    output_path = config.get_output_path('predictions.csv')
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    print("\n" + "="*80)
    print("CP3 COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {output_path}")


if __name__ == '__main__':
    main()
