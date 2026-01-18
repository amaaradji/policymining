#!/usr/bin/env python3
"""
CP2 Enhanced: Inject labeled violations WITH NOISE (Checkpoint 2).

Extends violation injection to include realistic noise scenarios:
- Near-miss cases: Approvals at Delta boundary ± epsilon
- Multiple approvals: Junior then senior, or multiple in sequence
- Missing role attributes: Some events lack role information
- Timestamp jitter: Small random delays
- Out-of-order events: Occasional sequence violations

This makes the benchmark harder and more realistic.

Outputs:
- event_log.csv/xes (modified log with violations AND noise)
- ground_truth.csv (case-level labels)
- noise_report.txt (summary of noise scenarios injected)
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


class NoisyViolationInjector:
    """Inject controlled violations AND noise scenarios into event logs."""

    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.T = config.senior_approval_threshold
        self.delta_hours = config.delegation_window_hours
        random.seed(config.seed)

        # Track noise statistics
        self.noise_stats = {
            'near_miss_cases': 0,
            'multiple_approval_cases': 0,
            'missing_role_events': 0,
            'jittered_events': 0,
            'out_of_order_pairs': 0
        }

    def inject_violations(self, log: EventLog) -> Tuple[EventLog, pd.DataFrame]:
        """
        Inject violations AND noise into clean log.

        Returns:
            (modified_log, ground_truth_df)
        """
        modified_log = EventLog()
        ground_truth_rows = []

        total_cases = len(log)
        cases_requiring_senior = sum(
            1 for trace in log
            if trace.attributes.get('amount', 0) >= self.T
        )

        target_violations = int(cases_requiring_senior * self.config.violation_rate)
        violations_injected = 0

        print(f"Total cases: {total_cases}")
        print(f"Cases requiring senior approval: {cases_requiring_senior}")
        print(f"Target violations: {target_violations}")

        for case_num, trace in enumerate(log, 1):
            case_id = trace.attributes['concept:name']
            amount = trace.attributes['amount']
            requires_senior = amount >= self.T

            # Decide violation injection
            inject_violation = False
            if requires_senior and violations_injected < target_violations:
                if violations_injected < target_violations * 0.5:
                    inject_violation = True
                else:
                    inject_violation = random.random() < 0.8

            if inject_violation:
                modified_trace, outcome = self._inject_violation_in_trace(trace, requires_senior)
                violations_injected += 1
            else:
                modified_trace, outcome = self._keep_trace_conforming(trace, requires_senior)

            # Apply noise scenarios (independent of violation)
            if self.config.enable_noise:
                modified_trace = self._apply_noise_scenarios(modified_trace, outcome)

            modified_log.append(modified_trace)

            ground_truth_rows.append({
                'case_id': case_id,
                'amount': amount,
                'requires_senior': requires_senior,
                'outcome': outcome,
                'is_violation': outcome == 'duty_unmet'
            })

            if case_num % 100 == 0:
                print(f"  Processed {case_num}/{total_cases}, violations: {violations_injected}")

        print(f"Final violations injected: {violations_injected}/{target_violations}")

        ground_truth_df = pd.DataFrame(ground_truth_rows)
        return modified_log, ground_truth_df

    def _inject_violation_in_trace(self, trace: Trace, requires_senior: bool) -> Tuple[Trace, str]:
        """Inject a violation into a single trace."""
        new_trace = Trace()
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
            approval_idx = None

            for i, event in enumerate(trace):
                new_event = Event(event)
                activity = event['concept:name']

                if activity in ['Senior_Approval', 'Junior_Approval']:
                    new_event['time:timestamp'] = event['time:timestamp'] + delay
                    approval_idx = len(new_trace)
                elif approval_idx is not None:
                    # Delay all events after approval
                    new_event['time:timestamp'] = event['time:timestamp'] + delay

                new_trace.append(new_event)

            outcome = 'duty_unmet'

        else:  # replace
            # Replace senior with junior
            for event in trace:
                new_event = Event(event)
                activity = event['concept:name']

                if activity == 'Senior_Approval':
                    new_event['concept:name'] = 'Junior_Approval'
                    new_event['role'] = 'junior'
                    new_event['org:resource'] = f'Junior_{random.randint(1, 3)}'

                new_trace.append(new_event)

            outcome = 'duty_met_via_delegation'

        return new_trace, outcome

    def _keep_trace_conforming(self, trace: Trace, requires_senior: bool) -> Tuple[Trace, str]:
        """Keep trace conforming (no violation)."""
        new_trace = Trace()
        for key, value in trace.attributes.items():
            new_trace.attributes[key] = value

        for event in trace:
            new_trace.append(Event(event))

        if not requires_senior:
            outcome = 'not_applicable'
        else:
            has_senior = any(e['concept:name'] == 'Senior_Approval' for e in trace)
            has_junior = any(e['concept:name'] == 'Junior_Approval' for e in trace)

            if has_senior:
                outcome = 'duty_met'
            elif has_junior:
                outcome = 'duty_met_via_delegation'
            else:
                outcome = 'duty_unmet'

        return new_trace, outcome

    def _apply_noise_scenarios(self, trace: Trace, outcome: str) -> Trace:
        """
        Apply various noise scenarios to make evaluation harder.

        Scenarios:
        1. Near-miss: Move approval to boundary ± epsilon
        2. Multiple approvals: Add extra approval events
        3. Missing role: Remove role attribute from some events
        4. Timestamp jitter: Add small random delays
        5. Out-of-order: Swap adjacent events occasionally
        """
        noisy_trace = Trace()
        for key, value in trace.attributes.items():
            noisy_trace.attributes[key] = value

        # Collect events
        events = [Event(e) for e in trace]

        # 1. Near-miss scenario (move approval to boundary)
        if random.random() < self.config.near_miss_rate:
            for event in events:
                if event['concept:name'] in ['Senior_Approval', 'Junior_Approval']:
                    # Find submit request time
                    submit_event = next((e for e in events if e['concept:name'] == 'Submit_Request'), None)
                    if submit_event:
                        # Move to delta boundary ± epsilon
                        delta = timedelta(hours=self.delta_hours)
                        epsilon = timedelta(minutes=random.randint(-self.config.boundary_epsilon_minutes,
                                                                   self.config.boundary_epsilon_minutes))
                        new_time = submit_event['time:timestamp'] + delta + epsilon
                        event['time:timestamp'] = new_time
                        self.noise_stats['near_miss_cases'] += 1
                        break  # Only modify one approval

        # 2. Multiple approvals scenario
        if random.random() < self.config.multiple_approval_rate:
            # Find approval event and duplicate it
            approval_idx = next((i for i, e in enumerate(events)
                                if e['concept:name'] in ['Senior_Approval', 'Junior_Approval']), None)
            if approval_idx is not None:
                # Add second approval slightly later
                second_approval = Event(events[approval_idx])
                second_approval['time:timestamp'] = events[approval_idx]['time:timestamp'] + timedelta(minutes=30)
                # Alternate role if possible
                if events[approval_idx]['concept:name'] == 'Junior_Approval':
                    second_approval['concept:name'] = 'Senior_Approval'
                    second_approval['role'] = 'senior'
                    second_approval['org:resource'] = f'Senior_{random.randint(1, 2)}'
                events.insert(approval_idx + 1, second_approval)
                self.noise_stats['multiple_approval_cases'] += 1

        # 3. Missing role attributes
        for event in events:
            if random.random() < self.config.missing_role_rate:
                if 'role' in event:
                    del event['role']
                    self.noise_stats['missing_role_events'] += 1

        # 4. Timestamp jitter
        for event in events:
            if random.random() < 0.3:  # Apply jitter to 30% of events
                jitter = timedelta(minutes=random.randint(-self.config.timestamp_jitter_minutes,
                                                          self.config.timestamp_jitter_minutes))
                event['time:timestamp'] = event['time:timestamp'] + jitter
                self.noise_stats['jittered_events'] += 1

        # 5. Out-of-order events (swap adjacent pairs)
        if random.random() < self.config.out_of_order_rate and len(events) > 2:
            # Pick random adjacent pair to swap
            idx = random.randint(1, len(events) - 2)
            events[idx], events[idx + 1] = events[idx + 1], events[idx]
            self.noise_stats['out_of_order_pairs'] += 1

        # Sort by timestamp to maintain some order (but jitter/swaps create noise)
        events.sort(key=lambda e: e['time:timestamp'])

        for event in events:
            noisy_trace.append(event)

        return noisy_trace

    def save_noise_report(self, config: SyntheticConfig):
        """Save summary of noise scenarios injected."""
        report_path = config.get_output_path('noise_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("NOISE INJECTION REPORT (Checkpoint 2)\n")
            f.write("="*80 + "\n\n")
            f.write("Noise scenarios applied to make evaluation more realistic:\n\n")
            f.write(f"Near-miss cases (approval at boundary ± {config.boundary_epsilon_minutes}min):\n")
            f.write(f"  Count: {self.noise_stats['near_miss_cases']}\n\n")
            f.write(f"Multiple approval cases:\n")
            f.write(f"  Count: {self.noise_stats['multiple_approval_cases']}\n\n")
            f.write(f"Missing role attributes:\n")
            f.write(f"  Count: {self.noise_stats['missing_role_events']} events\n\n")
            f.write(f"Timestamp jitter (±{config.timestamp_jitter_minutes}min):\n")
            f.write(f"  Count: {self.noise_stats['jittered_events']} events\n\n")
            f.write(f"Out-of-order event pairs:\n")
            f.write(f"  Count: {self.noise_stats['out_of_order_pairs']} pairs\n\n")
            f.write("These noise scenarios test the robustness of the policy checker and\n")
            f.write("prevent 'testing code against itself' by introducing realistic variations.\n")

        print(f"Saved noise report to: {report_path}")


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
                'role': event.get('role', ''),  # May be missing due to noise
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


def print_statistics(ground_truth_df: pd.DataFrame, log_df: pd.DataFrame,
                    noise_stats: Dict):
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

    print("\n" + "="*80)
    print("NOISE STATISTICS")
    print("="*80)
    print(f"Near-miss cases:            {noise_stats['near_miss_cases']}")
    print(f"Multiple approval cases:    {noise_stats['multiple_approval_cases']}")
    print(f"Missing role events:        {noise_stats['missing_role_events']}")
    print(f"Jittered events:            {noise_stats['jittered_events']}")
    print(f"Out-of-order pairs:         {noise_stats['out_of_order_pairs']}")


def main():
    """Main execution for CP2 Noisy."""
    print("="*80)
    print("CP2 NOISY: INJECT VIOLATIONS WITH REALISTIC NOISE")
    print("="*80)

    # Find latest run (should have CP1 output)
    runs = sorted(Path('eval/synthetic/outputs').glob('*/'))
    if not runs:
        print("ERROR: No CP1 output found. Run CP1 first.")
        return

    latest_run = runs[-1]
    run_id = latest_run.name

    config = SyntheticConfig(run_id=run_id)
    print(f"\nConfiguration:")
    print(f"  Run ID: {config.run_id}")
    print(f"  Noise enabled: {config.enable_noise}")
    print(f"  Near-miss rate: {config.near_miss_rate*100:.1f}%")
    print(f"  Multiple approval rate: {config.multiple_approval_rate*100:.1f}%")
    print(f"  Missing role rate: {config.missing_role_rate*100:.1f}%")

    # Load clean log from CP1
    print("\n" + "="*80)
    print("LOADING CLEAN LOG (CP1)")
    print("="*80)
    clean_log = load_clean_log(config)
    print(f"Loaded {len(clean_log)} cases")

    # Inject violations AND noise
    print("\n" + "="*80)
    print("INJECTING VIOLATIONS AND NOISE")
    print("="*80)
    injector = NoisyViolationInjector(config)
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

    # Save noise report
    print("\n" + "="*80)
    print("SAVING NOISE REPORT")
    print("="*80)
    injector.save_noise_report(config)

    # Print statistics
    print_statistics(ground_truth_df, log_df, injector.noise_stats)

    print("\n" + "="*80)
    print("CP2 NOISY COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {config.get_output_path('event_log.xes')} (modified with noise)")
    print(f"  - {config.get_output_path('event_log.csv')} (modified with noise)")
    print(f"  - {config.get_output_path('ground_truth.csv')}")
    print(f"  - {config.get_output_path('noise_report.txt')}")


if __name__ == '__main__':
    main()
