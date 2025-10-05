#!/usr/bin/env python3
"""
Generate policy log for P1 (Senior Approval Duty) on BPI 2017.
Policy-only conformance checking (no alignment with process model).

Usage:
    python generate_policy_log_p1.py [--config config.yaml] [--synthetic]
"""

import argparse
import json
import re
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from utils_io import (
    load_event_log,
    extract_case_amounts,
    build_role_mapping,
    create_synthetic_sample
)


def load_config(config_path: str) -> dict:
    """Load policy configuration from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['policy_config']


def find_request_anchor(case_df: pd.DataFrame, request_anchors: List[str]) -> Optional[pd.Timestamp]:
    """
    Find timestamp of first request anchor event in case.

    Args:
        case_df: DataFrame for single case (sorted by timestamp)
        request_anchors: List of activity names that serve as request anchors

    Returns:
        Timestamp of first anchor event, or None if not found
    """
    anchor_df = case_df[case_df['activity'].isin(request_anchors)]
    if len(anchor_df) > 0:
        return anchor_df.iloc[0]['timestamp']
    return None


def find_approvals(case_df: pd.DataFrame, approval_acts: List[str],
                   role_map: Dict[str, str]) -> List[dict]:
    """
    Find all approval events with role classification.

    Args:
        case_df: DataFrame for single case
        approval_acts: List of activity names that represent approvals
        role_map: Mapping from resource name to role (senior/junior/unknown)

    Returns:
        List of approval event dictionaries with seq, ts, resource, role
    """
    approvals = []
    approval_df = case_df[case_df['activity'].isin(approval_acts)]

    for idx, row in approval_df.iterrows():
        resource = str(row['resource']) if pd.notna(row['resource']) else 'UNKNOWN'
        approvals.append({
            'seq': row['seq'],
            'ts': row['timestamp'],
            'resource': resource,
            'role': role_map.get(resource, 'unknown'),
            'activity': row['activity']
        })

    return approvals


def evaluate_p1_duty(case_id: str, target_event: dict, amount: float,
                     request_ts: Optional[pd.Timestamp], approvals: List[dict],
                     config: dict) -> dict:
    """
    Evaluate P1 duty (senior approval requirement) for a single target event.

    Policy Logic:
    - If amount >= T: senior approval is required (duty)
    - If no senior approval within delta_hours: junior approval allowed (delegation)
    - Otherwise: duty_unmet (violation)

    Args:
        case_id: Case identifier
        target_event: Dictionary with seq, activity, ts, resource
        amount: Loan amount for this case
        request_ts: Timestamp of request anchor (A_Submitted)
        approvals: List of approval events with role classification
        config: Policy configuration dictionary

    Returns:
        Dictionary representing one row in the policy log
    """
    T = config['T']
    delta = timedelta(hours=config['delta_hours'])
    event_ts = target_event['ts']

    requires_senior = (amount >= T)
    wait_hours = (event_ts - request_ts).total_seconds() / 3600 if request_ts else None

    # Filter approvals that occurred before or at target event timestamp
    all_approvals_before_target = [a for a in approvals if a['ts'] <= event_ts]

    # Filter approvals within delta window from request anchor
    approvals_within_delta = []
    if request_ts is not None:
        approvals_within_delta = [a for a in all_approvals_before_target
                                  if a['ts'] <= request_ts + delta]
    else:
        approvals_within_delta = all_approvals_before_target

    # Find senior approval within delta window
    senior_approval = next((a for a in approvals_within_delta if a.get('role') == 'senior'), None)

    # Find junior approval - check within delta first, then outside delta for delegation
    junior_approval_within = next((a for a in approvals_within_delta if a.get('role') == 'junior'), None)
    junior_approval_outside = next((a for a in all_approvals_before_target
                                    if a.get('role') == 'junior' and
                                    (request_ts is None or a['ts'] > request_ts + delta)), None)

    # Determine outcome based on policy logic
    if not requires_senior:
        outcome = 'not_applicable'
        evidence = f'amount={amount:.2f} < T={T} (policy does not apply)'
        senior_seq = None
        junior_seq = None

    elif senior_approval is not None:
        # Senior approved within delta - duty met
        outcome = 'duty_met'
        activity = str(senior_approval.get('activity', 'UNKNOWN'))
        resource = str(senior_approval.get('resource', 'UNKNOWN'))
        seq = senior_approval.get('seq', -1)
        wait_str = f"{wait_hours:.1f}" if wait_hours is not None else "N/A"
        evidence = f'Senior approval: {activity} by {resource} at seq {seq} (wait={wait_str}h)'
        senior_seq = seq if isinstance(seq, (int, float)) and seq >= 0 else None
        junior_seq = None

    elif junior_approval_within is not None:
        # No senior within delta, but junior within delta - delegation allowed
        outcome = 'duty_met_via_delegation'
        activity = str(junior_approval_within.get('activity', 'UNKNOWN'))
        resource = str(junior_approval_within.get('resource', 'UNKNOWN'))
        seq = junior_approval_within.get('seq', -1)
        wait_str = f"{wait_hours:.1f}" if wait_hours is not None else "N/A"
        evidence = f'Junior approval (delegation): {activity} by {resource} at seq {seq} (wait={wait_str}h, no senior within {config["delta_hours"]}h)'
        senior_seq = None
        junior_seq = seq if isinstance(seq, (int, float)) and seq >= 0 else None

    elif junior_approval_outside is not None:
        # No approval within delta, but junior exists outside - late delegation
        outcome = 'duty_met_via_delegation'
        activity = str(junior_approval_outside.get('activity', 'UNKNOWN'))
        resource = str(junior_approval_outside.get('resource', 'UNKNOWN'))
        seq = junior_approval_outside.get('seq', -1)
        ts_diff = (junior_approval_outside['ts'] - request_ts).total_seconds() / 3600 if request_ts else 0
        evidence = f'Junior approval (late delegation): {activity} by {resource} at seq {seq} ({ts_diff:.1f}h after request, beyond {config["delta_hours"]}h window)'
        senior_seq = None
        junior_seq = seq if isinstance(seq, (int, float)) and seq >= 0 else None

    else:
        # No approval at all
        outcome = 'duty_unmet'
        if wait_hours is not None:
            evidence = (f'VIOLATION: No approval found '
                       f'(wait={wait_hours:.1f}h, amount={amount:.2f} >= T={T})')
        else:
            evidence = f'VIOLATION: No request anchor found, no approval detected'
        senior_seq = None
        junior_seq = None

    return {
        'case_id': case_id,
        'seq': target_event['seq'],
        'event_activity': target_event['activity'],
        'event_ts': event_ts,
        'performer': target_event['resource'],
        'policy_id': config['policy_id'],
        'rule_id': 'senior_approval_duty',
        'amount': amount,
        'T': T,
        'requires_senior': requires_senior,
        'request_ts': request_ts,
        'wait_hours': wait_hours,
        'senior_approval_seq': senior_seq,
        'junior_approval_seq': junior_seq,
        'outcome': outcome,
        'evidence': evidence
    }


def generate_policy_log(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Main pipeline to generate policy log from event log.

    Args:
        df: Event log DataFrame
        config: Policy configuration dictionary

    Returns:
        Policy log DataFrame
    """
    # Add sequence numbers if missing (per-case sequencing)
    if 'seq' not in df.columns:
        df = df.sort_values(['case_id', 'timestamp']).reset_index(drop=True)
        df['seq'] = df.groupby('case_id').cumcount() + 1

    # Extract case-level amounts
    case_amounts = extract_case_amounts(df)

    # Build role mapping (with approval activity fallback)
    role_map = build_role_mapping(df, config['senior_regex'], config['approval_acts'])

    print(f"\nRole classification summary:")
    role_counts = Counter(role_map.values())
    for role, count in role_counts.items():
        print(f"  {role}: {count} resources")

    policy_rows = []

    # Process each case
    for case_id in df['case_id'].unique():
        case_df = df[df['case_id'] == case_id].sort_values('timestamp')
        amount = case_amounts.get(case_id, 0.0)

        # Find request anchor timestamp
        request_ts = find_request_anchor(case_df, config['request_anchors'])

        # Find all approvals with role classification
        approvals = find_approvals(case_df, config['approval_acts'], role_map)

        # Find target events to evaluate
        target_events = case_df[case_df['activity'].isin(config['target_acts'])]

        for idx, row in target_events.iterrows():
            target_event = {
                'seq': row['seq'],
                'activity': row['activity'],
                'ts': row['timestamp'],
                'resource': str(row['resource']) if pd.notna(row['resource']) else 'UNKNOWN'
            }

            policy_row = evaluate_p1_duty(
                case_id, target_event, amount, request_ts, approvals, config
            )
            policy_rows.append(policy_row)

    # Create DataFrame
    policy_df = pd.DataFrame(policy_rows)

    return policy_df


def print_summary(policy_df: pd.DataFrame):
    """Print console summary of policy log."""
    print("\n" + "="*80)
    print("POLICY LOG GENERATION SUMMARY (P1: Senior Approval Duty)")
    print("="*80)

    print(f"\nTotal evaluations: {len(policy_df)}")

    # Outcome counts
    print("\nOutcome distribution:")
    outcome_counts = policy_df['outcome'].value_counts()
    for outcome, count in outcome_counts.items():
        pct = 100 * count / len(policy_df)
        print(f"  {outcome:30s}: {count:5d} ({pct:5.1f}%)")

    # Cases with violations
    cases_with_violations = policy_df[policy_df['outcome'] == 'duty_unmet']['case_id'].nunique()
    total_cases = policy_df['case_id'].nunique()
    violation_pct = 100 * cases_with_violations / total_cases if total_cases > 0 else 0
    print(f"\nCases with at least one duty_unmet: {cases_with_violations}/{total_cases} ({violation_pct:.1f}%)")

    # Violation rate (event-level)
    violation_rate = (policy_df['outcome'] == 'duty_unmet').sum() / len(policy_df)
    print(f"Event-level violation rate: {violation_rate:.3f}")

    # Top 5 evidence patterns
    print("\nTop 5 evidence patterns:")
    evidence_patterns = Counter()
    for evidence in policy_df['evidence']:
        # Extract pattern (first 60 chars or until specific numbers)
        pattern = re.sub(r'seq \d+', 'seq N', str(evidence))
        pattern = re.sub(r'wait=[\d.]+h', 'wait=Xh', pattern)
        pattern = re.sub(r'amount=[\d.]+', 'amount=X', pattern)
        pattern = pattern[:80]
        evidence_patterns[pattern] += 1

    for i, (pattern, count) in enumerate(evidence_patterns.most_common(5), 1):
        print(f"  {i}. [{count:4d}x] {pattern}")

    print("\n" + "="*80)


def save_summary(policy_df: pd.DataFrame, output_dir: Path):
    """Save summary statistics to JSON."""
    outcome_counts = policy_df['outcome'].value_counts().to_dict()
    cases_with_violations = policy_df[policy_df['outcome'] == 'duty_unmet']['case_id'].nunique()
    total_cases = policy_df['case_id'].nunique()
    violation_rate = (policy_df['outcome'] == 'duty_unmet').sum() / len(policy_df)

    summary = {
        'total_evaluations': len(policy_df),
        'total_cases': int(total_cases),
        'outcomes': {k: int(v) for k, v in outcome_counts.items()},
        'cases_with_violations': int(cases_with_violations),
        'violation_rate_event_level': float(violation_rate),
        'violation_rate_case_level': float(cases_with_violations / total_cases if total_cases > 0 else 0),
        'requires_senior_count': int(policy_df['requires_senior'].sum()),
        'avg_wait_hours': float(policy_df['wait_hours'].mean()) if policy_df['wait_hours'].notna().any() else None
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate P1 policy log')
    parser.add_argument('--config', default='eval/policy_only/config.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic 2-case sample for testing')
    parser.add_argument('--max-cases', type=int, default=None,
                       help='Limit to first N cases (for testing)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Policy: {config['policy_id']}")
    print(f"Threshold T: {config['T']}")
    print(f"Delegation window: {config['delta_hours']} hours")

    # Load event log
    if args.synthetic:
        print("\nUsing synthetic 2-case sample for testing...")
        df = create_synthetic_sample()
    else:
        event_log_path = config['event_log_path']
        print(f"\nLoading event log from: {event_log_path}")
        df = load_event_log(event_log_path)

    print(f"Loaded {len(df)} events from {df['case_id'].nunique()} cases")

    # Limit to max_cases if specified
    if args.max_cases is not None:
        case_ids = df['case_id'].unique()[:args.max_cases]
        df = df[df['case_id'].isin(case_ids)]
        print(f"Limited to first {args.max_cases} cases ({len(df)} events)")

    # Generate policy log
    print("\nGenerating policy log...")
    policy_df = generate_policy_log(df, config)

    # Save policy log
    output_path = output_dir / 'policy_log.csv'
    policy_df.to_csv(output_path, index=False)
    print(f"\nPolicy log saved to: {output_path}")

    # Print summary
    print_summary(policy_df)

    # Save summary JSON
    save_summary(policy_df, output_dir)


if __name__ == '__main__':
    main()
