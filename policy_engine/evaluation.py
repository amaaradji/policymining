#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation Framework for Policy Engine

This module provides utilities for evaluating policy checking performance,
including synthetic violation injection and comparison of different approaches.

Author: Manus AI
Date: October 2025
"""

import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """
    Evaluation framework for policy checking approaches.

    Supports:
    - Synthetic violation injection
    - Performance metrics (precision, recall, F1)
    - Comparison of event-log-only vs policy-aware approaches
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the policy evaluator.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        self.violations_injected = []

    def inject_availability_violations(self,
                                       df_events: pd.DataFrame,
                                       violation_rate: float = 0.05,
                                       availability_config: Dict = None) -> pd.DataFrame:
        """
        Inject synthetic availability violations into event log.

        Args:
            df_events: DataFrame containing events
            violation_rate: Proportion of events to violate (default 5%)
            availability_config: Configuration for availability windows

        Returns:
            DataFrame with injected violations and ground truth labels
        """
        logger.info(f"Injecting availability violations (rate={violation_rate})...")

        # Copy dataframe to avoid modifying original
        df = df_events.copy()

        # Add ground truth column
        df['is_violation'] = False
        df['original_timestamp'] = df['timestamp']

        # Default availability config
        if availability_config is None:
            availability_config = {
                'start_hour': 9,
                'end_hour': 17,
                'allowed_days': [0, 1, 2, 3, 4]  # Monday-Friday
            }

        # Select events to violate
        num_violations = int(len(df) * violation_rate)
        violation_indices = random.sample(range(len(df)), num_violations)

        # Inject violations
        for idx in violation_indices:
            timestamp = df.loc[idx, 'timestamp']

            # Randomly choose violation type: outside hours or outside days
            if random.random() < 0.5:
                # Violate working hours (set to evening/night)
                new_hour = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21, 22, 23])
                new_timestamp = timestamp.replace(hour=new_hour, minute=random.randint(0, 59))
            else:
                # Violate working days (set to weekend)
                days_to_add = random.choice([1, 2])  # Move to next weekend
                new_timestamp = timestamp + timedelta(days=days_to_add)
                while new_timestamp.weekday() not in [5, 6]:  # Saturday or Sunday
                    new_timestamp += timedelta(days=1)

            # Update timestamp and mark as violation
            df.loc[idx, 'timestamp'] = new_timestamp
            df.loc[idx, 'is_violation'] = True

            # Record violation info
            self.violations_injected.append({
                'case_id': df.loc[idx, 'case_id'],
                'seq': df.loc[idx, 'seq'],
                'activity': df.loc[idx, 'activity'],
                'original_ts': timestamp,
                'violated_ts': new_timestamp
            })

        logger.info(f"Injected {num_violations} violations")

        return df

    def evaluate_detection(self,
                          df_with_ground_truth: pd.DataFrame,
                          policy_log_df: pd.DataFrame) -> Dict:
        """
        Evaluate detection performance against ground truth.

        Args:
            df_with_ground_truth: Events with is_violation column
            policy_log_df: Policy log with detected violations

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating detection performance...")

        # Merge ground truth with detections
        detected_violations = set()
        for _, row in policy_log_df.iterrows():
            if row['outcome'] == 'duty_unmet':
                detected_violations.add((row['case_id'], row['seq']))

        # Calculate metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        for _, row in df_with_ground_truth.iterrows():
            is_actual_violation = row['is_violation']
            is_detected = (row['case_id'], row['seq']) in detected_violations

            if is_actual_violation and is_detected:
                true_positives += 1
            elif is_actual_violation and not is_detected:
                false_negatives += 1
            elif not is_actual_violation and is_detected:
                false_positives += 1
            else:
                true_negatives += 1

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_actual_violations': true_positives + false_negatives,
            'total_detected_violations': true_positives + false_positives
        }

        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        return results

    def compare_approaches(self,
                          df_with_ground_truth: pd.DataFrame,
                          policy_aware_log: pd.DataFrame,
                          event_only_log: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compare policy-aware vs event-log-only approaches.

        Args:
            df_with_ground_truth: Events with ground truth violations
            policy_aware_log: Policy log from policy-aware approach
            event_only_log: Policy log from event-only approach (optional)

        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing detection approaches...")

        # Evaluate policy-aware approach
        policy_aware_results = self.evaluate_detection(df_with_ground_truth, policy_aware_log)

        results = [{
            'approach': 'Policy-Aware',
            **policy_aware_results
        }]

        # Evaluate event-only approach if provided
        if event_only_log is not None:
            event_only_results = self.evaluate_detection(df_with_ground_truth, event_only_log)
            results.append({
                'approach': 'Event-Log-Only',
                **event_only_results
            })

        return pd.DataFrame(results)

    def generate_summary_report(self, comparison_df: pd.DataFrame) -> str:
        """
        Generate a text summary report of evaluation results.

        Args:
            comparison_df: DataFrame with comparison results

        Returns:
            Formatted summary report string
        """
        report = []
        report.append("=" * 60)
        report.append("POLICY EVALUATION SUMMARY")
        report.append("=" * 60)
        report.append("")

        for _, row in comparison_df.iterrows():
            report.append(f"Approach: {row['approach']}")
            report.append("-" * 40)
            report.append(f"  Precision: {row['precision']:.4f}")
            report.append(f"  Recall:    {row['recall']:.4f}")
            report.append(f"  F1 Score:  {row['f1_score']:.4f}")
            report.append("")
            report.append(f"  True Positives:  {row['true_positives']}")
            report.append(f"  False Positives: {row['false_positives']}")
            report.append(f"  False Negatives: {row['false_negatives']}")
            report.append(f"  True Negatives:  {row['true_negatives']}")
            report.append("")
            report.append(f"  Total Actual Violations:   {row['total_actual_violations']}")
            report.append(f"  Total Detected Violations: {row['total_detected_violations']}")
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)
