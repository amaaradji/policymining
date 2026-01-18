#!/usr/bin/env python3
"""
CP1: Generate clean synthetic event log using PM4Py.

Creates a synthetic process model and simulates executions to produce:
- event_log.xes (XES format)
- event_log.csv (CSV format)

The log represents a simplified purchase order approval process with:
- Activities: Submit Request, Check Amount, Junior Approval, Senior Approval,
             Process Payment, Send Notification, Close Case
- Case attributes: amount (for policy evaluation)
- Event attributes: timestamp, resource, activity
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pm4py.objects.log.obj import EventLog, Event, Trace
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from config import SyntheticConfig


def generate_simple_process_log(config: SyntheticConfig) -> EventLog:
    """
    Generate a simple synthetic event log for a purchase approval process.

    Process flow:
    1. Submit Request
    2. Check Amount
    3. Junior Approval (for amounts < threshold) OR Senior Approval (for amounts >= threshold)
    4. Process Payment
    5. Send Notification
    6. Close Case

    Args:
        config: Configuration object with generation parameters

    Returns:
        PM4Py EventLog object
    """
    random.seed(config.seed)

    log = EventLog()

    # Define activity templates
    activities = [
        'Submit_Request',
        'Check_Amount',
        'Junior_Approval',
        'Senior_Approval',
        'Process_Payment',
        'Send_Notification',
        'Close_Case'
    ]

    # Resources
    clerks = [f'Clerk_{i}' for i in range(1, 6)]
    junior_approvers = [f'Junior_{i}' for i in range(1, 4)]
    senior_approvers = [f'Senior_{i}' for i in range(1, 3)]
    processors = [f'Processor_{i}' for i in range(1, 4)]

    print(f"Generating {config.num_cases} cases...")

    for case_num in range(1, config.num_cases + 1):
        trace = Trace()
        trace.attributes['concept:name'] = f'Case_{case_num}'

        # Generate case attributes
        # Amount follows a log-normal distribution to simulate realistic purchase amounts
        amount = random.lognormvariate(9.5, 0.8)  # Mean ~15K, with spread
        amount = round(amount, 2)
        trace.attributes['amount'] = amount

        # Determine if senior approval is required
        requires_senior = amount >= config.senior_approval_threshold

        # Generate timestamps
        start_time = datetime(2024, 1, 1) + timedelta(
            days=random.randint(0, 180),
            hours=random.randint(8, 16)
        )

        current_time = start_time

        # Activity 1: Submit Request
        event = Event()
        event['concept:name'] = 'Submit_Request'
        event['time:timestamp'] = current_time
        event['org:resource'] = random.choice(clerks)
        event['lifecycle:transition'] = 'complete'
        trace.append(event)
        current_time += timedelta(minutes=random.randint(5, 30))

        # Activity 2: Check Amount
        event = Event()
        event['concept:name'] = 'Check_Amount'
        event['time:timestamp'] = current_time
        event['org:resource'] = random.choice(clerks)
        event['lifecycle:transition'] = 'complete'
        trace.append(event)
        current_time += timedelta(minutes=random.randint(10, 60))

        # Activity 3: Approval (Junior or Senior based on amount)
        if requires_senior:
            # Senior approval required for high amounts
            event = Event()
            event['concept:name'] = 'Senior_Approval'
            event['time:timestamp'] = current_time
            event['org:resource'] = random.choice(senior_approvers)
            event['lifecycle:transition'] = 'complete'
            event['role'] = 'senior'
            trace.append(event)
            current_time += timedelta(hours=random.randint(1, 12))
        else:
            # Junior approval for lower amounts
            event = Event()
            event['concept:name'] = 'Junior_Approval'
            event['time:timestamp'] = current_time
            event['org:resource'] = random.choice(junior_approvers)
            event['lifecycle:transition'] = 'complete'
            event['role'] = 'junior'
            trace.append(event)
            current_time += timedelta(hours=random.randint(1, 6))

        # Activity 4: Process Payment
        event = Event()
        event['concept:name'] = 'Process_Payment'
        event['time:timestamp'] = current_time
        event['org:resource'] = random.choice(processors)
        event['lifecycle:transition'] = 'complete'
        trace.append(event)
        current_time += timedelta(hours=random.randint(2, 24))

        # Activity 5: Send Notification
        event = Event()
        event['concept:name'] = 'Send_Notification'
        event['time:timestamp'] = current_time
        event['org:resource'] = random.choice(clerks)
        event['lifecycle:transition'] = 'complete'
        trace.append(event)
        current_time += timedelta(minutes=random.randint(5, 15))

        # Activity 6: Close Case
        event = Event()
        event['concept:name'] = 'Close_Case'
        event['time:timestamp'] = current_time
        event['org:resource'] = random.choice(clerks)
        event['lifecycle:transition'] = 'complete'
        trace.append(event)

        log.append(trace)

        if case_num % 100 == 0:
            print(f"  Generated {case_num}/{config.num_cases} cases")

    return log


def export_log_to_xes(log: EventLog, output_path: Path) -> None:
    """Export event log to XES format."""
    print(f"Exporting to XES: {output_path}")
    xes_exporter.apply(log, str(output_path))


def export_log_to_csv(log: EventLog, output_path: Path) -> None:
    """
    Export event log to CSV format.

    CSV schema:
    - case_id: case identifier
    - activity: activity name
    - timestamp: event timestamp
    - resource: resource who performed the activity
    - role: role of the resource (for approval activities)
    - amount: case attribute (amount)
    """
    print(f"Exporting to CSV: {output_path}")

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
    df.to_csv(output_path, index=False)

    return df


def print_log_statistics(log: EventLog, df: pd.DataFrame) -> None:
    """Print summary statistics about the generated log."""
    print("\n" + "="*80)
    print("LOG STATISTICS")
    print("="*80)

    num_cases = len(log)
    num_events = sum(len(trace) for trace in log)
    num_activities = len(df['activity'].unique())

    print(f"Total cases: {num_cases}")
    print(f"Total events: {num_events}")
    print(f"Unique activities: {num_activities}")
    print(f"Average events per case: {num_events/num_cases:.1f}")

    # Amount statistics
    amounts = [trace.attributes.get('amount', 0) for trace in log]
    print(f"\nAmount statistics:")
    print(f"  Mean: ${sum(amounts)/len(amounts):,.2f}")
    print(f"  Min: ${min(amounts):,.2f}")
    print(f"  Max: ${max(amounts):,.2f}")

    # Activity distribution
    print(f"\nActivity distribution:")
    activity_counts = df['activity'].value_counts()
    for activity, count in activity_counts.items():
        print(f"  {activity:25s}: {count:5d}")

    # Senior vs Junior approval counts
    senior_count = df[df['activity'] == 'Senior_Approval'].shape[0]
    junior_count = df[df['activity'] == 'Junior_Approval'].shape[0]
    print(f"\nApproval distribution:")
    print(f"  Senior Approval: {senior_count:5d} ({100*senior_count/num_cases:.1f}%)")
    print(f"  Junior Approval: {junior_count:5d} ({100*junior_count/num_cases:.1f}%)")


def main():
    """Main execution for CP1."""
    print("="*80)
    print("CP1: GENERATE CLEAN SYNTHETIC LOG")
    print("="*80)

    # Initialize configuration
    config = SyntheticConfig()
    print(f"\nConfiguration:")
    print(f"  Seed: {config.seed}")
    print(f"  Run ID: {config.run_id}")
    print(f"  Number of cases: {config.num_cases}")
    print(f"  Senior approval threshold: ${config.senior_approval_threshold:,.2f}")

    # Create output directory
    output_dir = config.create_output_dir()
    print(f"  Output directory: {output_dir}")

    # Generate synthetic log
    print("\n" + "="*80)
    print("GENERATING SYNTHETIC LOG")
    print("="*80)
    log = generate_simple_process_log(config)

    # Export to XES
    print("\n" + "="*80)
    print("EXPORTING TO XES")
    print("="*80)
    xes_path = config.get_output_path('event_log.xes')
    export_log_to_xes(log, xes_path)

    # Export to CSV
    print("\n" + "="*80)
    print("EXPORTING TO CSV")
    print("="*80)
    csv_path = config.get_output_path('event_log.csv')
    df = export_log_to_csv(log, csv_path)

    # Print statistics
    print_log_statistics(log, df)

    print("\n" + "="*80)
    print("CP1 COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {xes_path}")
    print(f"  - {csv_path}")


if __name__ == '__main__':
    main()
