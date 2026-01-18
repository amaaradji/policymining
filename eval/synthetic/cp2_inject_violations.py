#!/usr/bin/env python3
"""
CP2: Inject labeled policy violations into clean synthetic log.

Creates controlled violation scenarios with ground truth labels:
- duty_unmet: Remove/delay approval to cause violations
- duty_met: Keep senior approval within delta window
- duty_met_via_delegation: Junior approval within delta (senior required)
- not_applicable: Amount < threshold (no policy applies)

Outputs:
- event_log.csv (modified log with violations)
- event_log.xes (modified log with violations)
- ground_truth.csv (case-level labels)
"""

import random
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pm4py.objects.log.obj import EventLog, Event, Trace
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer

from config import SyntheticConfig


class ViolationInjector:
    """Inject controlled policy violations into event logs."""

    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.T = config.senior_approval_threshold
        self.delta_hours = config.delegation_window_hours
        random.seed(config.seed)

    def inject_violations(self, log: EventLog) -> Tuple[EventLog, pd.DataFrame]:
        """
        Inject violations into clean log and generate ground truth.

        Strategies for violation injection:
        1. Remove approval events (no approval scenario)
        2. Delay approval beyond delta window
        3. Replace senior with junior approval (delegation scenario)
        4. Keep some cases clean (conforming cases)

        Args:
            log: Clean EventLog from CP1

        Returns:
            (modified_log, ground_truth_df)
        """
        modified_log = EventLog()
        ground_truth_rows = []

        # Count cases by amount category
        total_cases = len(log)
        cases_requiring_senior = sum(
            1 for trace in log
            if trace.attributes.get('amount', 0) >= self.T
        )

        # Calculate target violations based on config
        target_violations = int(cases_requiring_senior * self.config.violation_rate)
        violations_injected = 0

        print(f"Total cases: {total_cases}")
        print(f"Cases requiring senior approval: {cases_requiring_senior}")
        print(f"Target violations to inject: {target_violations} ({self.config.violation_rate*100:.1f}%)")

        for case_num, trace in enumerate(log, 1):
            case_id = trace.attributes['concept:name']
            amount = trace.attributes['amount']
            requires_senior = amount >= self.T

            # Decide whether to inject violation
            inject_violation = False
            if requires_senior and violations_injected < target_violations:
                # Inject violations in a controlled manner
                # First 50% get violations deterministically, rest randomly
                if violations_injected < target_violations * 0.5:
                    inject_violation = True
                else:
                    inject_violation = random.random() < 0.8

            if inject_violation:
                modified_trace, outcome = self._inject_violation_in_trace(trace, requires_senior)
                violations_injected += 1
            else:
                modified_trace, outcome = self._keep_trace_conforming(trace, requires_senior)

            modified_log.append(modified_trace)

            # Record ground truth
            ground_truth_rows.append({
                'case_id': case_id,
                'amount': amount,
                'requires_senior': requires_senior,
                'outcome': outcome,
                'is_violation': outcome == 'duty_unmet'
            })

            if case_num % 100 == 0:
                print(f"  Processed {case_num}/{total_cases} cases, violations injected: {violations_injected}")

        print(f"Final violations injected: {violations_injected}/{target_violations}")

        ground_truth_df = pd.DataFrame(ground_truth_rows)
        return modified_log, ground_truth_df

    def _inject_violation_in_trace(self, trace: Trace, requires_senior: bool) -> Tuple[Trace, str]:
        """
        Inject a violation into a single trace.

        Violation strategies:
        1. Remove approval event entirely (40% of violations)
        2. Delay approval beyond delta window (30% of violations)
        3. Replace senior with junior outside delta (30% of violations)

        Args:
            trace: Original trace
            requires_senior: Whether policy applies to this case

        Returns:
            (modified_trace, outcome)
        """
        new_trace = Trace()
        # Copy attributes manually
        for key, value in trace.attributes.items():
            new_trace.attributes[key] = value

        violation_type = random.choice(['remove', 'delay', 'replace'])

        if violation_type == 'remove':
            # Remove approval event entirely
            for event in trace:
                activity = event['concept:name']
                if activity not in ['Senior_Approval', 'Junior_Approval']:
                    new_trace.append(event)
            outcome = 'duty_unmet'

        elif violation_type == 'delay':
            # Delay approval beyond delta window
            delay = timedelta(hours=self.delta_hours + random.randint(12, 72))

            for event in trace:
                activity = event['concept:name']
                new_event = Event(event)  # Copy event

                if activity in ['Senior_Approval', 'Junior_Approval']:
                    # Delay this approval and all subsequent events
                    new_event['time:timestamp'] = event['time:timestamp'] + delay
                    # Also need to delay all events after this
                    approval_idx = len(new_trace)

                new_trace.append(new_event)

            # Delay all events after approval
            if violation_type == 'delay':
                for i in range(approval_idx + 1, len(new_trace)):
                    new_trace[i]['time:timestamp'] = new_trace[i]['time:timestamp'] + delay

            outcome = 'duty_unmet'

        else:  # replace
            # Replace senior with junior approval, keep timing
            for event in trace:
                activity = event['concept:name']
                new_event = Event(event)

                if activity == 'Senior_Approval':
                    # Replace with junior approval
                    new_event['concept:name'] = 'Junior_Approval'
                    new_event['role'] = 'junior'
                    new_event['org:resource'] = f'Junior_{random.randint(1, 3)}'

                new_trace.append(new_event)

            outcome = 'duty_met_via_delegation'

        return new_trace, outcome

    def _keep_trace_conforming(self, trace: Trace, requires_senior: bool) -> Tuple[Trace, str]:
        """
        Keep trace conforming (no violation).

        Args:
            trace: Original trace
            requires_senior: Whether policy applies

        Returns:
            (trace, outcome)
        """
        if not requires_senior:
            outcome = 'not_applicable'
        else:
            # Check what approval exists in original trace
            has_senior = any(e['concept:name'] == 'Senior_Approval' for e in trace)
            has_junior = any(e['concept:name'] == 'Junior_Approval' for e in trace)

            if has_senior:
                outcome = 'duty_met'
            elif has_junior:
                outcome = 'duty_met_via_delegation'
            else:
                # Should not happen in clean log, but handle it
                outcome = 'duty_unmet'

        return trace, outcome


def load_clean_log(config: SyntheticConfig) -> EventLog:
    """Load the clean log from CP1."""
    xes_path = config.get_output_path('event_log.xes')
    print(f"Loading clean log from: {xes_path}")
    log = xes_importer.apply(str(xes_path))
    return log


def export_modified_log(log: EventLog, config: SyntheticConfig) -> pd.DataFrame:
    """Export modified log to XES and CSV."""
    # Export XES
    xes_path = config.get_output_path('event_log.xes')
    print(f"Exporting modified log to XES: {xes_path}")
    xes_exporter.apply(log, str(xes_path))

    # Export CSV
    csv_path = config.get_output_path('event_log.csv')
    print(f"Exporting modified log to CSV: {csv_path}")

    rows = []
    for trace in log:
        case_id = trace.attributes['concept:name']
        amount = trace.attributes.get('amount', 0)

        for event in trace:
            row = {
                'case_id': case_id,
                'activity': event['concept:name'],
                'timestamp': event['time:timestamp'],
                'resource': event.get('org:resource', ''),
                'role': event.get('role', ''),
                'amount': amount
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(['case_id', 'timestamp'])
    df.to_csv(csv_path, index=False)

    return df


def save_ground_truth(ground_truth_df: pd.DataFrame, config: SyntheticConfig):
    """Save ground truth labels to CSV."""
    output_path = config.get_output_path('ground_truth.csv')
    print(f"Saving ground truth to: {output_path}")
    ground_truth_df.to_csv(output_path, index=False)


def print_statistics(ground_truth_df: pd.DataFrame, log_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("GROUND TRUTH STATISTICS")
    print("="*80)

    total_cases = len(ground_truth_df)
    print(f"Total cases: {total_cases}")

    # Outcome distribution
    print("\nOutcome distribution:")
    outcome_counts = ground_truth_df['outcome'].value_counts()
    for outcome in ['not_applicable', 'duty_met', 'duty_met_via_delegation', 'duty_unmet']:
        count = outcome_counts.get(outcome, 0)
        pct = 100 * count / total_cases
        print(f"  {outcome:30s}: {count:5d} ({pct:5.1f}%)")

    # Violation statistics
    violations = ground_truth_df['is_violation'].sum()
    applicable_cases = ground_truth_df[ground_truth_df['requires_senior']].shape[0]
    violation_rate = violations / applicable_cases if applicable_cases > 0 else 0

    print(f"\nViolation statistics:")
    print(f"  Total violations: {violations}")
    print(f"  Applicable cases: {applicable_cases}")
    print(f"  Violation rate (among applicable): {100*violation_rate:.1f}%")

    # Activity distribution in modified log
    print(f"\nActivity distribution (modified log):")
    activity_counts = log_df['activity'].value_counts()
    for activity, count in activity_counts.items():
        print(f"  {activity:25s}: {count:5d}")


def main():
    """Main execution for CP2."""
    print("="*80)
    print("CP2: INJECT LABELED VIOLATIONS")
    print("="*80)

    # Load configuration (use same run_id as CP1)
    # For demo purposes, we'll use the latest run
    import os
    runs = sorted(Path('eval/synthetic/outputs').glob('*/'))
    if not runs:
        print("ERROR: No CP1 output found. Run CP1 first.")
        return

    latest_run = runs[-1]
    run_id = latest_run.name

    config = SyntheticConfig(run_id=run_id)
    print(f"\nConfiguration:")
    print(f"  Run ID: {config.run_id}")
    print(f"  Seed: {config.seed}")
    print(f"  Senior approval threshold: ${config.senior_approval_threshold:,.2f}")
    print(f"  Delegation window: {config.delegation_window_hours} hours")
    print(f"  Violation injection rate: {config.violation_rate*100:.1f}%")

    # Load clean log from CP1
    print("\n" + "="*80)
    print("LOADING CLEAN LOG (CP1)")
    print("="*80)
    clean_log = load_clean_log(config)
    print(f"Loaded {len(clean_log)} cases")

    # Inject violations
    print("\n" + "="*80)
    print("INJECTING VIOLATIONS")
    print("="*80)
    injector = ViolationInjector(config)
    modified_log, ground_truth_df = injector.inject_violations(clean_log)

    # Export modified log
    print("\n" + "="*80)
    print("EXPORTING MODIFIED LOG")
    print("="*80)
    log_df = export_modified_log(modified_log, config)

    # Save ground truth
    print("\n" + "="*80)
    print("SAVING GROUND TRUTH")
    print("="*80)
    save_ground_truth(ground_truth_df, config)

    # Print statistics
    print_statistics(ground_truth_df, log_df)

    print("\n" + "="*80)
    print("CP2 COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {config.get_output_path('event_log.xes')} (modified)")
    print(f"  - {config.get_output_path('event_log.csv')} (modified)")
    print(f"  - {config.get_output_path('ground_truth.csv')}")


if __name__ == '__main__':
    main()
