#!/usr/bin/env python3
"""
I/O utilities for policy log generation.
Handles event log loading, role mapping, and data extraction.
"""

import pandas as pd
import pm4py
from typing import Dict, Optional
from pathlib import Path


def load_event_log(file_path: str) -> pd.DataFrame:
    """
    Load event log from XES or CSV file (with CSV caching for XES).

    Args:
        file_path: Path to event log file (.xes or .csv)

    Returns:
        DataFrame with standardized column names
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Event log not found: {file_path}")

    if file_path.suffix.lower() == '.xes':
        # Check for cached CSV version
        cache_path = file_path.with_suffix('.cached.csv')
        if cache_path.exists():
            print(f"  Loading from cache: {cache_path}")
            df = pd.read_csv(cache_path)
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='mixed', utc=True)
        else:
            print(f"  Parsing XES (this may take a few minutes)...")
            log = pm4py.read_xes(str(file_path), show_progress_bar=True)
            df = pm4py.convert_to_dataframe(log)
            # Save cache
            print(f"  Saving cache to: {cache_path}")
            df.to_csv(cache_path, index=False)
    elif file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
        # Try to parse timestamp column
        for col in ['time:timestamp', 'timestamp', 'time', 'Complete Timestamp']:
            if col in df.columns:
                df['time:timestamp'] = pd.to_datetime(df[col], utc=True)
                break
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Standardize column names
    column_mapping = {
        'case:concept:name': 'case_id',
        'concept:name': 'activity',
        'org:resource': 'resource',
        'time:timestamp': 'timestamp'
    }

    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Ensure timestamp is datetime with UTC
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    return df


def extract_case_amounts(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract loan amount per case from case or event attributes.

    Args:
        df: Event log DataFrame

    Returns:
        Dictionary mapping case_id to amount
    """
    case_amounts = {}

    # Possible column names for amount in BPI 2017
    amount_cols = [
        'case:RequestedAmount',
        'RequestedAmount',
        'case:ApplicationType',
        'ApplicationType',
        'case:LoanGoal',
        'LoanGoal',
        'amount'
    ]

    # Group by case
    for case_id in df['case_id'].unique():
        case_df = df[df['case_id'] == case_id]

        # Try to find amount attribute
        amount = None
        for col in amount_cols:
            if col in case_df.columns:
                val = case_df[col].iloc[0]
                if pd.notna(val):
                    try:
                        amount = float(val)
                        break
                    except (ValueError, TypeError):
                        continue

        # Default to 0 if not found
        case_amounts[case_id] = amount if amount is not None else 0.0

    return case_amounts


def build_role_mapping(df: pd.DataFrame, senior_regex: str, approval_acts: list = None) -> Dict[str, str]:
    """
    Classify resources as senior/junior/unknown using hybrid approach:
    1. First try regex pattern matching
    2. If no seniors found, use approval activity patterns to assign senior roles

    Senior assignment criteria (if regex fails):
    - Top 30% most active approvers = senior
    - Resources who approved high-value cases (>50k) = senior
    - Ensures realistic senior presence in the data

    Args:
        df: Event log DataFrame with 'resource' column
        senior_regex: Regular expression to identify senior resources
        approval_acts: List of approval activity names (for pattern-based assignment)

    Returns:
        Dictionary mapping resource name to role (senior/junior/unknown)
    """
    import re

    role_map = {}
    pattern = re.compile(senior_regex)

    # First attempt: regex-based classification
    for resource in df['resource'].unique():
        if pd.isna(resource) or resource == '' or resource == 'UNKNOWN':
            role_map[str(resource)] = 'unknown'
        elif pattern.search(str(resource)):
            role_map[str(resource)] = 'senior'
        else:
            role_map[str(resource)] = 'junior'

    # Check if any seniors were found
    senior_count = sum(1 for role in role_map.values() if role == 'senior')

    if senior_count == 0 and approval_acts:
        print("  No seniors found via regex. Using approval pattern-based assignment...")

        # Get approval activities
        approval_df = df[df['activity'].isin(approval_acts)].copy()

        if len(approval_df) > 0:
            # Count approvals per resource
            approval_counts = approval_df['resource'].value_counts()

            # Get amount data if available
            amount_col = None
            for col in ['case:RequestedAmount', 'RequestedAmount', 'amount']:
                if col in df.columns:
                    amount_col = col
                    break

            # Identify senior candidates
            senior_candidates = set()

            # Criterion 1: Top 30% most active approvers
            top_30_pct = int(len(approval_counts) * 0.3)
            if top_30_pct > 0:
                top_approvers = approval_counts.head(top_30_pct).index.tolist()
                senior_candidates.update(top_approvers)

            # Criterion 2: High-value case approvers (if amount data available)
            if amount_col:
                high_value_threshold = 50000
                high_value_cases = df[df[amount_col] > high_value_threshold]['case_id'].unique()
                if len(high_value_cases) > 0:
                    high_value_approvers = approval_df[
                        approval_df['case_id'].isin(high_value_cases)
                    ]['resource'].unique()
                    senior_candidates.update(high_value_approvers)

            # Update role map
            for resource in senior_candidates:
                if str(resource) in role_map and role_map[str(resource)] != 'unknown':
                    role_map[str(resource)] = 'senior'

            senior_count = sum(1 for role in role_map.values() if role == 'senior')
            print(f"  Assigned {senior_count} senior roles based on approval patterns")

    return role_map


def load_role_mapping_from_file(file_path: str) -> Optional[Dict[str, str]]:
    """
    Load role mapping from external CSV file (optional).

    Expected format:
    resource,role
    User_1,senior
    User_2,junior

    Args:
        file_path: Path to role mapping CSV

    Returns:
        Dictionary mapping resource to role, or None if file not found
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)
    return dict(zip(df['resource'], df['role']))


def create_synthetic_sample() -> pd.DataFrame:
    """
    Create a tiny synthetic 2-case sample for testing the pipeline.

    Returns:
        DataFrame with synthetic event log data
    """
    from datetime import datetime, timedelta

    base_time = datetime(2025, 1, 1, 9, 0, 0)

    events = [
        # Case 1: High amount, senior approval within 24h -> duty_met
        {'case_id': 'Case_1', 'activity': 'A_Submitted', 'resource': 'User_1',
         'timestamp': base_time, 'amount': 25000},
        {'case_id': 'Case_1', 'activity': 'W_Validate application', 'resource': 'SENIOR_User_2',
         'timestamp': base_time + timedelta(hours=10), 'amount': 25000},
        {'case_id': 'Case_1', 'activity': 'O_Accepted', 'resource': 'System',
         'timestamp': base_time + timedelta(hours=12), 'amount': 25000},

        # Case 2: High amount, no senior, junior delegation after 30h -> duty_met_via_delegation
        {'case_id': 'Case_2', 'activity': 'A_Submitted', 'resource': 'User_3',
         'timestamp': base_time, 'amount': 30000},
        {'case_id': 'Case_2', 'activity': 'W_Approve offer', 'resource': 'User_4',
         'timestamp': base_time + timedelta(hours=20), 'amount': 30000},
        {'case_id': 'Case_2', 'activity': 'A_Approved', 'resource': 'System',
         'timestamp': base_time + timedelta(hours=22), 'amount': 30000},

        # Case 3: Low amount -> not_applicable
        {'case_id': 'Case_3', 'activity': 'A_Submitted', 'resource': 'User_5',
         'timestamp': base_time, 'amount': 15000},
        {'case_id': 'Case_3', 'activity': 'O_Accepted', 'resource': 'System',
         'timestamp': base_time + timedelta(hours=5), 'amount': 15000},

        # Case 4: High amount, no approval within delta -> duty_unmet
        {'case_id': 'Case_4', 'activity': 'A_Submitted', 'resource': 'User_6',
         'timestamp': base_time, 'amount': 40000},
        {'case_id': 'Case_4', 'activity': 'A_Approved', 'resource': 'System',
         'timestamp': base_time + timedelta(hours=30), 'amount': 40000},
    ]

    df = pd.DataFrame(events)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    return df
