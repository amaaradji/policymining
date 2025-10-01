#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Policy Engine for Business Process Conformance Checking

This module implements a general, policy-only conformance checking framework
for business processes. It supports multiple policy types and produces
a unified policy log output.

Author: Manus AI
Date: October 2025
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Policy(ABC):
    """
    Abstract base class for policy definitions.
    
    All specific policy implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, policy_id: str, config: Dict):
        """
        Initialize the policy.
        
        Args:
            policy_id: Unique identifier for the policy
            config: Configuration dictionary with policy parameters
        """
        self.policy_id = policy_id
        self.config = config
        self.unknown_role_count = 0
    
    @abstractmethod
    def prepare_case(self, case_df: pd.DataFrame, context: Dict) -> Dict:
        """
        Precompute case-level information needed for policy evaluation.
        
        Args:
            case_df: DataFrame containing events for a single case
            context: Additional context information
            
        Returns:
            Dictionary with precomputed case state
        """
        pass
    
    @abstractmethod
    def evaluate_event(self, event_row: pd.Series, case_state: Dict, context: Dict) -> Optional[Dict]:
        """
        Evaluate a single event against the policy.
        
        Args:
            event_row: Series containing event data
            case_state: Precomputed case state from prepare_case
            context: Additional context information
            
        Returns:
            Dictionary with policy log row data, or None if event is not a target
        """
        pass


class SeniorApprovalPolicy(Policy):
    """
    Implementation of Policy P1: Senior approval if amount ≥ T; delegation allowed after 24h.
    """
    
    def __init__(self, policy_id: str, config: Dict):
        """
        Initialize the senior approval policy.
        
        Args:
            policy_id: Unique identifier for the policy
            config: Configuration dictionary with policy parameters
        """
        super().__init__(policy_id, config)
        self.threshold = config['thresholds'].get('P1_T', 20000)
        self.delegate_wait_hours = config['thresholds'].get('P1_delegate_wait_h', 24)
        self.target_acts = set(config['activities'].get('TARGET_ACTS', []))
        self.approval_acts = set(config['activities'].get('APPROVAL_ACTS', []))
        self.request_anchors = set(config['activities'].get('REQUEST_ANCHORS', []))
        self.senior_regex = re.compile(config['roles'].get('senior_regex', '(?i)(SENIOR|MANAGER)'))
        self.unknown_role_is_junior = config['defaults'].get('unknown_role_is_junior', True)
        
        logger.info(f"Initialized {policy_id} with threshold={self.threshold}, "
                   f"delegate_wait_hours={self.delegate_wait_hours}")
    
    def is_senior_role(self, role: str) -> bool:
        """
        Check if a role is considered senior.
        
        Args:
            role: Role string to check
            
        Returns:
            True if role matches senior pattern, False otherwise
        """
        if pd.isna(role) or not role:
            self.unknown_role_count += 1
            return not self.unknown_role_is_junior
        
        return bool(self.senior_regex.search(str(role)))
    
    def prepare_case(self, case_df: pd.DataFrame, context: Dict) -> Dict:
        """
        Precompute case-level information needed for policy evaluation.
        
        Args:
            case_df: DataFrame containing events for a single case
            context: Additional context information
            
        Returns:
            Dictionary with precomputed case state
        """
        # Get case amount (first non-null value)
        amount = None
        if 'amount' in case_df.columns:
            amount_values = case_df['amount'].dropna()
            if not amount_values.empty:
                amount = float(amount_values.iloc[0])
        
        # Determine if senior approval is required
        requires_senior = amount is not None and amount >= self.threshold
        
        # Find request timestamp (first event matching request anchors)
        request_ts = None
        request_events = case_df[case_df['activity'].isin(self.request_anchors)]
        if not request_events.empty:
            request_ts = request_events['timestamp'].min()
        else:
            # Fallback: use first event timestamp
            request_ts = case_df['timestamp'].min()
        
        # Build approvals table
        approvals = case_df[case_df['activity'].isin(self.approval_acts)].copy()
        if not approvals.empty:
            approvals['is_senior'] = approvals['role'].apply(self.is_senior_role)
        
        return {
            'amount': amount,
            'requires_senior': requires_senior,
            'request_ts': request_ts,
            'approvals': approvals
        }
    
    def evaluate_event(self, event_row: pd.Series, case_state: Dict, context: Dict) -> Optional[Dict]:
        """
        Evaluate a single event against the policy.
        
        Args:
            event_row: Series containing event data
            case_state: Precomputed case state from prepare_case
            context: Additional context information
            
        Returns:
            Dictionary with policy log row data, or None if event is not a target
        """
        # Skip if not a target activity
        if event_row['activity'] not in self.target_acts:
            return None
        
        # Extract case state
        amount = case_state.get('amount')
        requires_senior = case_state.get('requires_senior', False)
        request_ts = case_state.get('request_ts')
        approvals = case_state.get('approvals', pd.DataFrame())
        
        # Calculate wait hours
        event_ts = event_row['timestamp']
        wait_hours = 0
        if request_ts is not None:
            wait_hours = (event_ts - request_ts).total_seconds() / 3600.0
        
        # Find approvals before this event
        event_seq = event_row['seq']
        approvals_before = approvals[approvals['seq'] <= event_seq].copy() if not approvals.empty else pd.DataFrame()
        
        # Check for senior and junior approvals
        senior_before = False
        junior_before = False
        senior_approval_seq = ""
        junior_approval_seq = ""
        
        if not approvals_before.empty:
            senior_before = approvals_before['is_senior'].any()
            junior_before = len(approvals_before) > 0 and not senior_before
            
            if senior_before and 'seq' in approvals_before.columns:
                senior_rows = approvals_before[approvals_before['is_senior']]
                if not senior_rows.empty:
                    senior_approval_seq = str(int(senior_rows['seq'].max()))
            
            if junior_before and 'seq' in approvals_before.columns:
                junior_rows = approvals_before[~approvals_before['is_senior']]
                if not junior_rows.empty:
                    junior_approval_seq = str(int(junior_rows['seq'].max()))
        
        # Determine outcome
        if not requires_senior:
            outcome = "not_applicable"
        elif senior_before:
            outcome = "duty_met"
        elif junior_before and wait_hours >= self.delegate_wait_hours:
            outcome = "duty_met_via_delegation"
        else:
            outcome = "duty_unmet"
        
        # Create evidence string
        evidence = f"senior={senior_before};junior={junior_before};wait={round(wait_hours, 1)}h"
        
        # Return policy log row
        return {
            'case_id': event_row['case_id'],
            'seq': int(event_row['seq']),
            'event_activity': event_row['activity'],
            'event_ts': event_row['timestamp'].isoformat(),
            'performer': event_row.get('performer', ''),
            'policy_id': self.policy_id,
            'rule_id': f"{self.policy_id}.senior_required",
            'amount': float(amount) if amount is not None else "",
            'T': self.threshold,
            'requires_senior': requires_senior,
            'request_ts': request_ts.isoformat() if request_ts is not None else "",
            'wait_hours': round(wait_hours, 2),
            'senior_approval_seq': senior_approval_seq,
            'junior_approval_seq': junior_approval_seq,
            'outcome': outcome,
            'evidence': evidence
        }


class ResourceAvailabilityPolicy(Policy):
    """
    Implementation of Policy P2: Resource availability constraints.
    Resources must work within their defined availability windows (working hours/days).
    """

    def __init__(self, policy_id: str, config: Dict):
        """
        Initialize the resource availability policy.

        Args:
            policy_id: Unique identifier for the policy
            config: Configuration dictionary with policy parameters
        """
        super().__init__(policy_id, config)
        self.target_acts = set(config['activities'].get('TARGET_ACTS', []))
        self.default_start_hour = config['availability'].get('default_start_hour', 9)
        self.default_end_hour = config['availability'].get('default_end_hour', 17)
        self.default_days = set(config['availability'].get('default_days', [0, 1, 2, 3, 4]))

        # Resource-specific availability windows (can be extended)
        self.resource_windows = config['availability'].get('resource_windows', {})

        logger.info(f"Initialized {policy_id} with default hours={self.default_start_hour}-{self.default_end_hour}, "
                   f"days={self.default_days}")

    def get_availability_window(self, resource: str) -> Dict:
        """
        Get availability window for a resource.

        Args:
            resource: Resource identifier

        Returns:
            Dictionary with start_hour, end_hour, and allowed_days
        """
        if resource in self.resource_windows:
            return self.resource_windows[resource]

        return {
            'start_hour': self.default_start_hour,
            'end_hour': self.default_end_hour,
            'allowed_days': self.default_days
        }

    def is_within_availability(self, timestamp: datetime, resource: str) -> Tuple[bool, str]:
        """
        Check if timestamp is within resource's availability window.

        Args:
            timestamp: Event timestamp
            resource: Resource identifier

        Returns:
            Tuple of (is_available, reason)
        """
        window = self.get_availability_window(resource)

        # Check day of week (0=Monday, 6=Sunday)
        day_of_week = timestamp.weekday()
        if day_of_week not in window['allowed_days']:
            return False, f"outside_allowed_days(day={day_of_week})"

        # Check hour of day
        hour = timestamp.hour
        if hour < window['start_hour'] or hour >= window['end_hour']:
            return False, f"outside_working_hours(hour={hour})"

        return True, "within_availability"

    def prepare_case(self, case_df: pd.DataFrame, context: Dict) -> Dict:
        """
        Precompute case-level information needed for policy evaluation.

        Args:
            case_df: DataFrame containing events for a single case
            context: Additional context information

        Returns:
            Dictionary with precomputed case state
        """
        # No case-level state needed for availability checking
        return {}

    def evaluate_event(self, event_row: pd.Series, case_state: Dict, context: Dict) -> Optional[Dict]:
        """
        Evaluate a single event against the policy.

        Args:
            event_row: Series containing event data
            case_state: Precomputed case state from prepare_case
            context: Additional context information

        Returns:
            Dictionary with policy log row data, or None if event is not a target
        """
        # Check all activities (availability applies to all events)
        # If target_acts is specified and not empty, only check those activities
        if self.target_acts and event_row['activity'] not in self.target_acts:
            return None

        timestamp = event_row['timestamp']
        resource = event_row.get('performer', '')

        # Check availability
        is_available, reason = self.is_within_availability(timestamp, resource)

        # Determine outcome
        outcome = "duty_met" if is_available else "duty_unmet"

        # Get availability window for evidence
        window = self.get_availability_window(resource)
        evidence = (f"day={timestamp.weekday()};hour={timestamp.hour};"
                   f"window={window['start_hour']}-{window['end_hour']};"
                   f"days={sorted(window['allowed_days'])};reason={reason}")

        # Return policy log row
        return {
            'case_id': event_row['case_id'],
            'seq': int(event_row['seq']),
            'event_activity': event_row['activity'],
            'event_ts': timestamp.isoformat(),
            'performer': resource,
            'policy_id': self.policy_id,
            'rule_id': f"{self.policy_id}.availability_window",
            'timestamp_day': timestamp.weekday(),
            'timestamp_hour': timestamp.hour,
            'allowed_days': str(sorted(window['allowed_days'])),
            'start_hour': window['start_hour'],
            'end_hour': window['end_hour'],
            'outcome': outcome,
            'evidence': evidence
        }


class PolicyEngine:
    """
    Main policy engine for conformance checking.
    
    This class orchestrates the policy evaluation process across multiple
    policies and produces a unified policy log output.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the policy engine.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.policies = self._initialize_policies()
        self.unknown_role_count = 0
        
        logger.info(f"Initialized PolicyEngine with {len(self.policies)} policies")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def _initialize_policies(self) -> List[Policy]:
        """
        Initialize policy objects based on configuration.

        Returns:
            List of initialized policy objects
        """
        policies = []

        # Get enabled policies from configuration
        enabled_policies = self.config.get('enabled_policies', ['P1'])

        # Initialize P1: Senior Approval Policy
        if 'P1' in enabled_policies:
            policies.append(SeniorApprovalPolicy("P1", self.config))

        # Initialize P2: Resource Availability Policy
        if 'P2' in enabled_policies:
            policies.append(ResourceAvailabilityPolicy("P2", self.config))

        # Additional policies can be added here

        return policies
    
    def load_events(self, events_path: str) -> pd.DataFrame:
        """
        Load events from CSV file.
        
        Args:
            events_path: Path to the events CSV file
            
        Returns:
            DataFrame containing events
        """
        logger.info(f"Loading events from {events_path}")
        
        # Determine file extension
        _, ext = os.path.splitext(events_path)
        
        if ext.lower() == '.csv':
            df = pd.read_csv(events_path)
        elif ext.lower() == '.xes':
            try:
                import pm4py
                log = pm4py.read_xes(events_path)
                df = pm4py.convert_to_dataframe(log)
            except ImportError:
                logger.error("PM4Py is required to read XES files. Please install it with 'pip install pm4py'")
                sys.exit(1)
        else:
            logger.error(f"Unsupported file format: {ext}")
            sys.exit(1)
        
        # Rename columns if needed
        column_mapping = {
            'case:concept:name': 'case_id',
            'concept:name': 'activity',
            'time:timestamp': 'timestamp',
            'org:resource': 'performer'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Ensure required columns exist
        required_columns = ['case_id', 'activity', 'timestamp', 'performer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            sys.exit(1)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Sort by case_id and timestamp
        df = df.sort_values(['case_id', 'timestamp'])
        
        # Add sequence number
        df['seq'] = df.groupby('case_id').cumcount() + 1
        
        logger.info(f"Loaded {len(df)} events from {len(df['case_id'].unique())} cases")
        return df
    
    def load_roles(self, roles_path: str) -> pd.DataFrame:
        """
        Load resource roles from CSV file.
        
        Args:
            roles_path: Path to the roles CSV file
            
        Returns:
            DataFrame containing resource roles
        """
        if not roles_path:
            return None
        
        logger.info(f"Loading roles from {roles_path}")
        df_roles = pd.read_csv(roles_path)
        
        # Check for required columns
        if 'resource' not in df_roles.columns or 'role' not in df_roles.columns:
            logger.error("Roles file must contain 'resource' and 'role' columns")
            sys.exit(1)
        
        logger.info(f"Loaded {len(df_roles)} resource-role mappings")
        return df_roles
    
    def process_events(self, df_events: pd.DataFrame, df_roles: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Process events and generate policy log.
        
        Args:
            df_events: DataFrame containing events
            df_roles: DataFrame containing resource roles (optional)
            
        Returns:
            DataFrame containing policy log
        """
        # Join roles if needed
        if 'role' not in df_events.columns and df_roles is not None:
            logger.info("Joining roles with events")
            df_events = df_events.merge(df_roles, how='left', left_on='performer', right_on='resource')
        
        # Initialize context
        context = {
            'timezone': self.config['defaults'].get('timezone', 'UTC')
        }
        
        # Initialize policy log rows
        policy_log_rows = []
        
        # Process each case
        case_count = 0
        event_count = 0
        violation_count = 0
        
        for case_id, case_df in df_events.groupby('case_id'):
            case_count += 1
            event_count += len(case_df)
            
            if case_count % 100 == 0:
                logger.info(f"Processed {case_count} cases...")
            
            # Prepare case state for each policy
            case_states = {}
            for policy in self.policies:
                case_states[policy.policy_id] = policy.prepare_case(case_df, context)
            
            # Evaluate each event against each policy
            for _, event_row in case_df.iterrows():
                for policy in self.policies:
                    result = policy.evaluate_event(event_row, case_states[policy.policy_id], context)
                    if result:
                        policy_log_rows.append(result)
                        if result['outcome'] == 'duty_unmet':
                            violation_count += 1
        
        # Create policy log DataFrame
        policy_log_df = pd.DataFrame(policy_log_rows)
        
        # Log statistics
        logger.info(f"Processed {case_count} cases with {event_count} events")
        logger.info(f"Generated {len(policy_log_df)} policy log entries")
        logger.info(f"Found {violation_count} duty_unmet violations")
        
        # Check for unknown roles
        unknown_role_count = sum(policy.unknown_role_count for policy in self.policies)
        if unknown_role_count > 0:
            logger.warning(f"Found {unknown_role_count} events with unknown roles (treated as junior)")
        
        return policy_log_df
    
    def save_policy_log(self, policy_log_df: pd.DataFrame, output_path: str):
        """
        Save policy log to CSV file.
        
        Args:
            policy_log_df: DataFrame containing policy log
            output_path: Path to save the policy log CSV
        """
        logger.info(f"Saving policy log to {output_path}")
        policy_log_df.to_csv(output_path, index=False)
    
    def print_summary(self, policy_log_df: pd.DataFrame):
        """
        Print summary statistics to stdout.
        
        Args:
            policy_log_df: DataFrame containing policy log
        """
        print("\n" + "="*50)
        print("POLICY ENGINE SUMMARY")
        print("="*50)
        
        # Overall statistics
        case_count = len(policy_log_df['case_id'].unique()) if not policy_log_df.empty else 0
        entry_count = len(policy_log_df)
        
        print(f"\nProcessed {case_count} cases, generated {entry_count} policy log entries")
        
        # Outcome statistics by policy
        if not policy_log_df.empty:
            print("\nOutcome counts by policy:")
            policy_outcomes = policy_log_df.groupby(['policy_id', 'outcome']).size().unstack(fill_value=0)
            print(policy_outcomes)
            
            # Violation rate
            print("\nViolation rates:")
            for policy_id in policy_log_df['policy_id'].unique():
                policy_df = policy_log_df[policy_log_df['policy_id'] == policy_id]
                violation_count = len(policy_df[policy_df['outcome'] == 'duty_unmet'])
                violation_rate = violation_count / len(policy_df) if len(policy_df) > 0 else 0
                print(f"{policy_id}: {violation_count} violations ({violation_rate:.2%})")
            
            # Cases with any violation
            cases_with_violations = policy_log_df[policy_log_df['outcome'] == 'duty_unmet']['case_id'].nunique()
            case_violation_rate = cases_with_violations / case_count if case_count > 0 else 0
            print(f"\nCases with any violation: {cases_with_violations} ({case_violation_rate:.2%})")
        
        # Sanity checks
        print("\nSanity checks:")
        unknown_role_count = sum(policy.unknown_role_count for policy in self.policies)
        if unknown_role_count > 0:
            print(f"- WARNING: {unknown_role_count} events with unknown roles (treated as junior)")
        else:
            print("- All roles were successfully mapped")
        
        print("="*50)
    
    def run(self, events_path: str, roles_path: str, output_path: str):
        """
        Run the policy engine end-to-end.
        
        Args:
            events_path: Path to the events CSV file
            roles_path: Path to the roles CSV file (optional)
            output_path: Path to save the policy log CSV
        """
        # Load data
        df_events = self.load_events(events_path)
        df_roles = self.load_roles(roles_path) if roles_path else None
        
        # Process events
        policy_log_df = self.process_events(df_events, df_roles)
        
        # Save policy log
        self.save_policy_log(policy_log_df, output_path)
        
        # Print summary
        self.print_summary(policy_log_df)
        
        logger.info("Policy engine run completed successfully")


def main():
    """
    Main entry point for the policy engine CLI.
    """
    parser = argparse.ArgumentParser(description='Policy Engine for Business Process Conformance Checking')
    parser.add_argument('--events', required=True, help='Path to events CSV or XES file')
    parser.add_argument('--roles', help='Path to resource roles CSV file')
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    parser.add_argument('--out', required=True, help='Path to output policy log CSV file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation with synthetic violations')
    parser.add_argument('--violation-rate', type=float, default=0.05, help='Violation injection rate for evaluation (default: 0.05)')
    parser.add_argument('--eval-out', help='Path to save evaluation results (for --evaluate mode)')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Run policy engine
    engine = PolicyEngine(args.config)

    if args.evaluate:
        # Evaluation mode: inject violations and measure performance
        logger.info("Running in evaluation mode with synthetic violation injection")

        # Load events
        df_events = engine.load_events(args.events)
        df_roles = engine.load_roles(args.roles) if args.roles else None

        # Inject violations
        from evaluation import PolicyEvaluator
        evaluator = PolicyEvaluator()

        # Get availability config from engine config
        availability_config = engine.config.get('availability', {})
        df_with_violations = evaluator.inject_availability_violations(
            df_events,
            violation_rate=args.violation_rate,
            availability_config=availability_config
        )

        # Process events with injected violations
        policy_log_df = engine.process_events(df_with_violations, df_roles)

        # Evaluate detection performance
        results = evaluator.evaluate_detection(df_with_violations, policy_log_df)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame([{
            'approach': 'Policy-Aware',
            **results
        }])

        # Generate summary report
        summary = evaluator.generate_summary_report(comparison_df)
        print(summary)

        # Save results if output path specified
        if args.eval_out:
            comparison_df.to_csv(args.eval_out, index=False)
            logger.info(f"Saved evaluation results to {args.eval_out}")

        # Save policy log
        engine.save_policy_log(policy_log_df, args.out)

    else:
        # Normal mode: just run policy checking
        engine.run(args.events, args.roles, args.out)


if __name__ == "__main__":
    main()
