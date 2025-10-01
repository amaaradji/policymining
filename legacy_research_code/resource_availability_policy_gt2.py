#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resource Availability Policy Mining Framework with GT2 Ground Truth Definition

This module implements a policy-aware conformance checking framework for business process mining
with a focus on resource availability constraints. It uses the BPI2017 dataset to detect
violations of temporal resource availability policies.

This version uses GT2 (All policy violations according to formal definitions) as ground truth.

Author: Manus AI
Date: September 2025
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Any, Optional, Union
import gc  # Garbage collection for memory optimization

# PM4Py imports
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.statistics.traces.generic.log import case_statistics

# Scikit-learn imports for evaluation metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class ResourceAvailabilityPolicyMining:
    """
    A framework for policy mining and conformance checking in business processes,
    focusing on resource availability constraints.
    
    This class provides functionality to:
    1. Load and preprocess event logs
    2. Define resource availability policies
    3. Augment event logs with availability violations
    4. Generate policy logs with explicit availability constraints
    5. Detect violations using both event-log-only and policy-aware approaches
    6. Evaluate and visualize results
    """
    
    def __init__(self, log_path: str = None, output_dir: str = 'results'):
        """
        Initialize the ResourceAvailabilityPolicyMining framework.
        
        Args:
            log_path: Path to the event log file (.xes format)
            output_dir: Directory to save results and visualizations
        """
        self.log_path = log_path
        self.output_dir = output_dir
        self.event_log = None
        self.policy_log = None
        self.log_df = None
        self.resource_availability = {}  # Resource availability windows
        self.violation_info = {}  # Information about policy violations
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_event_log(self, log_path: str, max_traces: int = None) -> EventLog:
        """
        Load an event log from a file.
        
        Args:
            log_path: Path to the event log file (.xes format)
            max_traces: Maximum number of traces to load (None for all)
            
        Returns:
            The loaded event log
        """
        print(f"Loading event log from {log_path}...")
        parameters = {}
        if max_traces is not None:
            parameters['max_traces'] = max_traces
        
        self.event_log = xes_importer.apply(log_path, parameters=parameters)
        print(f"Loaded event log with {len(self.event_log)} traces")
        return self.event_log
    
    def preprocess_event_log(self) -> pd.DataFrame:
        """
        Preprocess the event log and convert to DataFrame for easier analysis.
        
        Returns:
            DataFrame containing the event log data
        """
        if self.event_log is None:
            raise ValueError("Event log not loaded. Call load_event_log first.")
        
        print("Preprocessing event log...")
        
        # Convert to DataFrame
        self.log_df = pm4py.convert_to_dataframe(self.event_log)
        
        # Free up memory
        self.event_log = None
        gc.collect()
        
        # Ensure timestamp is datetime
        if 'time:timestamp' in self.log_df.columns:
            self.log_df['time:timestamp'] = pd.to_datetime(self.log_df['time:timestamp'])
        
        print(f"Preprocessed event log with {len(self.log_df)} events")
        return self.log_df
    
    def define_resource_availability(self, method: str = 'fixed_hours') -> Dict[str, Dict]:
        """
        Define availability windows for each resource.
        
        Args:
            method: Method to define availability ('fixed_hours', 'shifts', or 'custom')
            
        Returns:
            Dictionary mapping resources to their availability windows
        """
        if self.log_df is None:
            self.preprocess_event_log()
        
        print(f"Defining resource availability using {method} method...")
        
        resources = self.log_df['org:resource'].unique()
        
        if method == 'fixed_hours':
            # Define standard working hours (9 AM - 5 PM) for all resources
            for resource in resources:
                self.resource_availability[resource] = {
                    'start_time': time(9, 0),  # 9:00 AM
                    'end_time': time(17, 0),   # 5:00 PM
                    'days': [0, 1, 2, 3, 4]    # Monday to Friday (0=Monday, 6=Sunday)
                }
        
        elif method == 'shifts':
            # Define different shifts for resources
            shifts = [
                {'start_time': time(6, 0), 'end_time': time(14, 0)},   # Morning shift: 6 AM - 2 PM
                {'start_time': time(14, 0), 'end_time': time(22, 0)},  # Afternoon shift: 2 PM - 10 PM
                {'start_time': time(22, 0), 'end_time': time(6, 0)}    # Night shift: 10 PM - 6 AM
            ]
            
            for i, resource in enumerate(resources):
                shift = shifts[i % len(shifts)]
                self.resource_availability[resource] = {
                    'start_time': shift['start_time'],
                    'end_time': shift['end_time'],
                    'days': [0, 1, 2, 3, 4, 5, 6]  # All days
                }
        
        elif method == 'custom':
            # Example of custom availability based on resource patterns
            # In a real scenario, this would be based on domain knowledge
            for resource in resources:
                # Randomly assign availability patterns
                hour_patterns = [
                    {'start_time': time(8, 0), 'end_time': time(16, 0)},
                    {'start_time': time(9, 0), 'end_time': time(17, 0)},
                    {'start_time': time(10, 0), 'end_time': time(18, 0)},
                    {'start_time': time(12, 0), 'end_time': time(20, 0)}
                ]
                day_patterns = [
                    [0, 1, 2, 3, 4],       # Monday to Friday
                    [0, 1, 2, 3],          # Monday to Thursday
                    [1, 2, 3, 4],          # Tuesday to Friday
                    [0, 2, 4]              # Monday, Wednesday, Friday
                ]
                
                pattern_idx = hash(resource) % len(hour_patterns)
                day_idx = hash(resource) % len(day_patterns)
                
                self.resource_availability[resource] = {
                    'start_time': hour_patterns[pattern_idx]['start_time'],
                    'end_time': hour_patterns[pattern_idx]['end_time'],
                    'days': day_patterns[day_idx]
                }
        
        else:
            raise ValueError(f"Unknown availability method: {method}")
        
        print(f"Defined availability windows for {len(self.resource_availability)} resources")
        return self.resource_availability
    
    def is_within_availability(self, resource: str, timestamp: datetime) -> bool:
        """
        Check if a timestamp is within a resource's availability window.
        
        Args:
            resource: Resource name
            timestamp: Timestamp to check
            
        Returns:
            True if timestamp is within availability window, False otherwise
        """
        if resource not in self.resource_availability:
            return True  # Default to available if no constraints defined
        
        availability = self.resource_availability[resource]
        
        # Check day of week (0=Monday, 6=Sunday)
        day_of_week = timestamp.weekday()
        if day_of_week not in availability['days']:
            return False
        
        # Check time of day
        start_time = availability['start_time']
        end_time = availability['end_time']
        current_time = timestamp.time()
        
        # Handle normal time range (e.g., 9 AM - 5 PM)
        if start_time < end_time:
            return start_time <= current_time < end_time
        # Handle overnight shifts (e.g., 10 PM - 6 AM)
        else:
            return current_time >= start_time or current_time < end_time
    
    def augment_event_log_with_violations(self, violation_rate: float = 0.05) -> pd.DataFrame:
        """
        Augment the event log with availability violations.
        
        Args:
            violation_rate: Percentage of events to modify as violations
            
        Returns:
            Augmented DataFrame with violations
        """
        if self.log_df is None:
            self.preprocess_event_log()
        
        if not self.resource_availability:
            self.define_resource_availability()
        
        print(f"Augmenting event log with availability violations (rate: {violation_rate:.2%})...")
        
        # Create a copy of the original DataFrame
        augmented_df = self.log_df.copy()
        
        # Add a column to track original timestamps
        augmented_df['original_timestamp'] = augmented_df['time:timestamp']
        
        # Calculate number of events to modify
        num_events = len(augmented_df)
        num_violations = int(num_events * violation_rate)
        
        # Randomly select events to modify
        violation_indices = random.sample(range(num_events), num_violations)
        
        # Track violation information for reporting
        self.violation_info = {
            'total_events': num_events,
            'injected_violations': num_violations,
            'violation_rate': violation_rate,
            'by_resource': {},
            'by_activity': {}
        }
        
        # Modify selected events to create violations
        for idx in violation_indices:
            row = augmented_df.iloc[idx]
            resource = row['org:resource']
            activity = row['concept:name']
            original_timestamp = row['time:timestamp']
            
            if resource not in self.resource_availability:
                continue
            
            availability = self.resource_availability[resource]
            
            # Create a violation by shifting the timestamp outside availability window
            new_timestamp = self.create_violation_timestamp(original_timestamp, availability)
            augmented_df.at[idx, 'time:timestamp'] = new_timestamp
            
            # Track violations by resource
            if resource not in self.violation_info['by_resource']:
                self.violation_info['by_resource'][resource] = 0
            self.violation_info['by_resource'][resource] += 1
            
            # Track violations by activity
            if activity not in self.violation_info['by_activity']:
                self.violation_info['by_activity'][activity] = 0
            self.violation_info['by_activity'][activity] += 1
        
        print(f"Augmented event log with {num_violations} injected violations")
        
        # Save the augmented log
        augmented_log_path = os.path.join(self.output_dir, 'augmented_event_log.csv')
        augmented_df.to_csv(augmented_log_path, index=False)
        print(f"Saved augmented event log to {augmented_log_path}")
        
        self.log_df = augmented_df
        return augmented_df
    
    def create_violation_timestamp(self, original_timestamp: datetime, availability: Dict) -> datetime:
        """
        Create a violation timestamp outside the availability window.
        
        Args:
            original_timestamp: Original timestamp
            availability: Resource availability window
            
        Returns:
            Modified timestamp that violates availability
        """
        day_of_week = original_timestamp.weekday()
        
        # If current day is not in availability days, keep the timestamp
        if day_of_week not in availability['days']:
            return original_timestamp
        
        start_time = availability['start_time']
        end_time = availability['end_time']
        current_time = original_timestamp.time()
        
        # Handle normal time range (e.g., 9 AM - 5 PM)
        if start_time < end_time:
            # If already outside availability, keep the timestamp
            if current_time < start_time or current_time >= end_time:
                return original_timestamp
            
            # Otherwise, shift to before or after availability
            if random.random() < 0.5:
                # Shift to before availability (early morning)
                hours_shift = -(current_time.hour - start_time.hour + 1 + random.randint(0, 2))
            else:
                # Shift to after availability (evening)
                hours_shift = end_time.hour - current_time.hour + random.randint(0, 2)
        
        # Handle overnight shifts (e.g., 10 PM - 6 AM)
        else:
            # If already outside availability, keep the timestamp
            if current_time >= start_time or current_time < end_time:
                return original_timestamp
            
            # Otherwise, shift to the middle of non-availability
            mid_hour = (end_time.hour + start_time.hour) // 2
            if mid_hour < end_time.hour:
                mid_hour += 12
            hours_shift = mid_hour - current_time.hour
        
        # Apply the shift
        return original_timestamp + timedelta(hours=hours_shift)
    
    def generate_policy_log(self) -> EventLog:
        """
        Generate a policy log with resource availability constraints.
        
        Returns:
            The generated policy log
        """
        if not self.resource_availability:
            self.define_resource_availability()
        
        print("Generating policy log with availability constraints...")
        
        # Create policy log
        policy_log = EventLog()
        
        # Add a trace for each resource with its availability policy
        for resource, availability in self.resource_availability.items():
            trace = Trace()
            trace.attributes['concept:name'] = resource
            
            # Format availability window for readability
            start_time_str = availability['start_time'].strftime('%H:%M')
            end_time_str = availability['end_time'].strftime('%H:%M')
            days_str = ','.join([str(d) for d in availability['days']])
            
            trace.attributes['availability_start'] = start_time_str
            trace.attributes['availability_end'] = end_time_str
            trace.attributes['availability_days'] = days_str
            
            # Add a policy definition event
            policy_event = Event()
            policy_event['concept:name'] = 'AvailabilityPolicyDefinition'
            policy_event['time:timestamp'] = datetime.now()
            policy_event['org:resource'] = resource
            policy_event['policy_type'] = 'availability'
            policy_event['start_time'] = start_time_str
            policy_event['end_time'] = end_time_str
            policy_event['days'] = days_str
            
            trace.append(policy_event)
            policy_log.append(trace)
        
        self.policy_log = policy_log
        print(f"Generated policy log with {len(policy_log)} resource availability policies")
        
        # Save the policy log
        policy_log_path = os.path.join(self.output_dir, 'availability_policy_log.xes')
        xes_exporter.apply(policy_log, policy_log_path)
        print(f"Saved policy log to {policy_log_path}")
        
        return policy_log
    
    def detect_violations_event_log_only(self) -> pd.DataFrame:
        """
        Detect availability violations using only the event log (baseline approach).
        
        This method uses statistical analysis to infer "normal" working hours
        from the event log itself, without using explicit policy definitions.
        
        Returns:
            DataFrame with detected violations
        """
        if self.log_df is None:
            raise ValueError("Event log not loaded or augmented.")
        
        print("Detecting violations using event log only (baseline approach)...")
        
        # Create a copy of the DataFrame for results
        result_df = self.log_df.copy()
        
        # Extract hour of day for each event
        result_df['hour_of_day'] = result_df['time:timestamp'].dt.hour
        result_df['day_of_week'] = result_df['time:timestamp'].dt.dayofweek
        
        # For each resource, infer "normal" working hours from the event log
        inferred_availability = {}
        
        for resource in result_df['org:resource'].unique():
            resource_events = result_df[result_df['org:resource'] == resource]
            
            # Skip resources with too few events
            if len(resource_events) < 5:
                continue
            
            # Infer working days (days with events)
            days_count = resource_events['day_of_week'].value_counts()
            active_days = days_count[days_count > 0].index.tolist()
            
            # Infer working hours (using histogram of event hours)
            hours_hist = resource_events['hour_of_day'].value_counts().sort_index()
            
            # Find continuous blocks of hours with events
            if len(hours_hist) < 2:
                continue
                
            # Use percentiles to determine working hours
            # This is a simple heuristic that could be improved
            hours = resource_events['hour_of_day'].values
            start_hour = int(np.percentile(hours, 5))
            end_hour = int(np.percentile(hours, 95))
            
            # Ensure we have a reasonable working window
            if end_hour <= start_hour:
                end_hour = start_hour + 8  # Assume 8-hour shift if inference fails
            
            inferred_availability[resource] = {
                'start_hour': start_hour,
                'end_hour': end_hour,
                'days': active_days
            }
        
        # Detect violations based on inferred availability
        result_df['eventlog_violation'] = False
        
        for idx, row in result_df.iterrows():
            resource = row['org:resource']
            timestamp = row['time:timestamp']
            hour = timestamp.hour
            day = timestamp.weekday()
            
            if resource in inferred_availability:
                avail = inferred_availability[resource]
                
                # Check if event is outside inferred availability
                if day not in avail['days'] or hour < avail['start_hour'] or hour >= avail['end_hour']:
                    result_df.at[idx, 'eventlog_violation'] = True
        
        # Count detected violations
        detected_violations = result_df['eventlog_violation'].sum()
        print(f"Event log only approach detected {detected_violations} violations")
        
        # Update the main DataFrame
        self.log_df['eventlog_violation'] = result_df['eventlog_violation']
        
        return result_df
    
    def detect_violations_policy_aware(self) -> pd.DataFrame:
        """
        Detect availability violations using policy logs (policy-aware approach).
        
        This method uses explicit policy definitions from the policy log
        to detect violations.
        
        Returns:
            DataFrame with detected violations
        """
        if self.log_df is None or not self.resource_availability:
            raise ValueError("Event log not loaded or resource availability not defined.")
        
        print("Detecting violations using policy-aware approach...")
        
        # Create a copy of the DataFrame for results
        result_df = self.log_df.copy()
        
        # Detect violations based on explicit policy definitions
        result_df['policy_violation'] = False
        
        # Count total policy violations for reporting
        total_policy_violations = 0
        
        for idx, row in result_df.iterrows():
            resource = row['org:resource']
            timestamp = row['time:timestamp']
            
            # Check if event is outside defined availability
            if not self.is_within_availability(resource, timestamp):
                result_df.at[idx, 'policy_violation'] = True
                total_policy_violations += 1
        
        # Update violation info with total policy violations
        self.violation_info['total_policy_violations'] = total_policy_violations
        
        # Count detected violations
        detected_violations = result_df['policy_violation'].sum()
        print(f"Policy-aware approach detected {detected_violations} violations")
        
        # Update the main DataFrame
        self.log_df['policy_violation'] = result_df['policy_violation']
        
        # Set ground truth based on policy violations (GT2)
        self.log_df['is_violation'] = self.log_df['policy_violation']
        
        return result_df
    
    def evaluate_detection_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the performance of violation detection methods.
        
        Returns:
            Dictionary with performance metrics for each method
        """
        if 'is_violation' not in self.log_df.columns:
            raise ValueError("Ground truth not defined. Run detect_violations_policy_aware first.")
        
        if 'eventlog_violation' not in self.log_df.columns or 'policy_violation' not in self.log_df.columns:
            raise ValueError("Violations not detected. Run detection methods first.")
        
        print("Evaluating detection performance...")
        
        # Calculate performance metrics for event log only method
        eventlog_precision = precision_score(self.log_df['is_violation'], self.log_df['eventlog_violation'])
        eventlog_recall = recall_score(self.log_df['is_violation'], self.log_df['eventlog_violation'])
        eventlog_f1 = f1_score(self.log_df['is_violation'], self.log_df['eventlog_violation'])
        
        # Calculate performance metrics for policy-aware method
        # Since we're using GT2, policy-aware method will have perfect metrics
        policy_precision = precision_score(self.log_df['is_violation'], self.log_df['policy_violation'])
        policy_recall = recall_score(self.log_df['is_violation'], self.log_df['policy_violation'])
        policy_f1 = f1_score(self.log_df['is_violation'], self.log_df['policy_violation'])
        
        # Calculate confusion matrices
        eventlog_cm = confusion_matrix(self.log_df['is_violation'], self.log_df['eventlog_violation'])
        policy_cm = confusion_matrix(self.log_df['is_violation'], self.log_df['policy_violation'])
        
        # Compile results
        results = {
            'event_log_only': {
                'precision': eventlog_precision,
                'recall': eventlog_recall,
                'f1': eventlog_f1,
                'confusion_matrix': eventlog_cm
            },
            'policy_aware': {
                'precision': policy_precision,
                'recall': policy_recall,
                'f1': policy_f1,
                'confusion_matrix': policy_cm
            }
        }
        
        print("\nPerformance Metrics (using GT2 - all policy violations as ground truth):")
        print(f"Event log only method: Precision={eventlog_precision:.4f}, Recall={eventlog_recall:.4f}, F1={eventlog_f1:.4f}")
        print(f"Policy-aware method: Precision={policy_precision:.4f}, Recall={policy_recall:.4f}, F1={policy_f1:.4f}")
        
        return results
    
    def visualize_results(self):
        """
        Visualize the results of availability violation detection.
        """
        if 'is_violation' not in self.log_df.columns:
            raise ValueError("Ground truth not defined. Run detect_violations_policy_aware first.")
        
        if 'eventlog_violation' not in self.log_df.columns or 'policy_violation' not in self.log_df.columns:
            raise ValueError("Violations not detected. Run detection methods first.")
        
        print("Visualizing results...")
        
        # Get performance metrics
        results = self.evaluate_detection_performance()
        eventlog_precision = results['event_log_only']['precision']
        eventlog_recall = results['event_log_only']['recall']
        eventlog_f1 = results['event_log_only']['f1']
        policy_precision = results['policy_aware']['precision']
        policy_recall = results['policy_aware']['recall']
        policy_f1 = results['policy_aware']['f1']
        eventlog_cm = results['event_log_only']['confusion_matrix']
        policy_cm = results['policy_aware']['confusion_matrix']
        
        # Create a figure with multiple subplots
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Distribution of events by hour of day
        plt.subplot(2, 2, 1)
        self.log_df['hour'] = self.log_df['time:timestamp'].dt.hour
        sns.histplot(data=self.log_df, x='hour', hue='is_violation', multiple='stack', bins=24)
        plt.title('Distribution of Events by Hour of Day', fontsize=14)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Number of Events', fontsize=12)
        plt.legend(['Normal', 'Violation'], fontsize=10)
        
        # Plot 2: Comparison of performance metrics
        plt.subplot(2, 2, 2)
        metrics = ['Precision', 'Recall', 'F1-Score']
        eventlog_scores = [eventlog_precision, eventlog_recall, eventlog_f1]
        policy_scores = [policy_precision, policy_recall, policy_f1]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, eventlog_scores, width, label='Event Log Only')
        plt.bar(x + width/2, policy_scores, width, label='Policy-Aware')
        plt.xticks(x, metrics, fontsize=12)
        plt.ylim(0, 1.1)
        plt.ylabel('Score', fontsize=12)
        plt.title('Detection Performance Comparison (GT2)', fontsize=14)
        plt.legend(fontsize=10)
        
        # Plot 3: Confusion Matrix for Event Log Only
        plt.subplot(2, 2, 3)
        sns.heatmap(eventlog_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Violation', 'Violation'],
                    yticklabels=['No Violation', 'Violation'])
        plt.title('Confusion Matrix: Event Log Only', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Plot 4: Confusion Matrix for Policy-Aware
        plt.subplot(2, 2, 4)
        sns.heatmap(policy_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Violation', 'Violation'],
                    yticklabels=['No Violation', 'Violation'])
        plt.title('Confusion Matrix: Policy-Aware', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'availability_violation_detection_results_gt2.png'), dpi=300)
        print(f"Saved detection results visualization to {os.path.join(self.output_dir, 'availability_violation_detection_results_gt2.png')}")
        
        # Free memory
        plt.close()
        gc.collect()
        
        # Create a separate figure for violations by resource
        plt.figure(figsize=(12, 8))
        
        # Count violations by resource
        resource_violations = self.log_df[self.log_df['is_violation']]['org:resource'].value_counts().reset_index()
        resource_violations.columns = ['Resource', 'Violations']
        
        # Plot top 15 resources by violation count
        top_n = min(15, len(resource_violations))
        resource_violations.head(top_n).plot(
            x='Resource', 
            y='Violations', 
            kind='bar', 
            color='crimson'
        )
        plt.title('Top Resources by Violation Count (GT2)', fontsize=14)
        plt.xlabel('Resource', fontsize=12)
        plt.ylabel('Number of Violations', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'violations_by_resource_gt2.png'), dpi=300)
        print(f"Saved violations by resource visualization to {os.path.join(self.output_dir, 'violations_by_resource_gt2.png')}")
        
        # Free memory
        plt.close()
        gc.collect()
        
        # Create summary results file
        with open(os.path.join(self.output_dir, 'availability_summary_results_gt2.txt'), 'w') as f:
            f.write('Resource Availability Violation Detection Results Summary (GT2)\n')
            f.write('==========================================================\n\n')
            f.write(f'Total events analyzed: {len(self.log_df)}\n')
            f.write(f'Total policy violations (GT2): {self.log_df["is_violation"].sum()} ({self.log_df["is_violation"].mean()*100:.2f}%)\n')
            f.write(f'Injected violations: {self.violation_info.get("injected_violations", "N/A")} ({self.violation_info.get("violation_rate", 0)*100:.2f}%)\n\n')
            
            f.write('Performance Metrics (using GT2 - all policy violations as ground truth):\n')
            f.write(f'Event log only method: Precision={eventlog_precision:.4f}, Recall={eventlog_recall:.4f}, F1={eventlog_f1:.4f}\n')
            f.write(f'Policy-aware method: Precision={policy_precision:.4f}, Recall={policy_recall:.4f}, F1={policy_f1:.4f}\n\n')
            
            f.write('Top 5 Resources by Violation Count:\n')
            for _, row in resource_violations.head(5).iterrows():
                f.write(f"  {row['Resource']}: {row['Violations']} violations\n")
            
            f.write('\nGround Truth Definition:\n')
            f.write('GT2: All policy violations according to formal definitions are considered true violations.\n')
            f.write('This approach provides a comprehensive evaluation of detection methods against actual policy compliance.\n')
        
        print(f"Saved summary results to {os.path.join(self.output_dir, 'availability_summary_results_gt2.txt')}")
    
    def save_results(self):
        """
        Save all results to files.
        """
        if self.log_df is None:
            raise ValueError("No results to save. Run detection methods first.")
        
        print("Saving all results...")
        
        # Save the analysis results
        self.log_df.to_csv(os.path.join(self.output_dir, 'availability_analysis_results_gt2.csv'), index=False)
        
        # Save violation information
        with open(os.path.join(self.output_dir, 'violation_info_gt2.txt'), 'w') as f:
            f.write('Violation Information (GT2)\n')
            f.write('========================\n\n')
            f.write(f'Total events: {len(self.log_df)}\n')
            f.write(f'Total policy violations: {self.log_df["is_violation"].sum()}\n')
            f.write(f'Policy violation rate: {self.log_df["is_violation"].mean()*100:.2f}%\n')
            if 'injected_violations' in self.violation_info:
                f.write(f'Injected violations: {self.violation_info["injected_violations"]}\n')
                f.write(f'Injection rate: {self.violation_info["violation_rate"]*100:.2f}%\n\n')
            
            f.write('\nGround Truth Definition:\n')
            f.write('GT2: All policy violations according to formal definitions are considered true violations.\n')
        
        print(f"All results saved to {self.output_dir}")
    
    def run_full_analysis(self, max_traces: int = None, violation_rate: float = 0.05):
        """
        Run the full analysis pipeline.
        
        Args:
            max_traces: Maximum number of traces to analyze (None for all)
            violation_rate: Percentage of events to modify as violations
        """
        print("Starting resource availability policy mining analysis with GT2 ground truth...")
        
        # Load event log if not already loaded
        if self.event_log is None and self.log_path:
            self.load_event_log(self.log_path, max_traces)
        
        # Preprocess event log
        self.preprocess_event_log()
        
        # Define resource availability
        self.define_resource_availability(method='fixed_hours')
        
        # Augment event log with violations
        self.augment_event_log_with_violations(violation_rate)
        
        # Generate policy log
        self.generate_policy_log()
        
        # Detect violations using policy-aware approach first to establish ground truth (GT2)
        self.detect_violations_policy_aware()
        
        # Detect violations using event log only approach
        self.detect_violations_event_log_only()
        
        # Evaluate detection performance
        self.evaluate_detection_performance()
        
        # Visualize results
        self.visualize_results()
        
        # Save all results
        self.save_results()
        
        print("Full analysis with GT2 ground truth completed successfully!")


def main():
    """
    Main function to run the resource availability policy mining framework.
    """
    # Set up directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Define paths
    log_path = 'BPI Challenge 2017.xes'
    output_dir = 'results'
    
    # Create framework instance
    framework = ResourceAvailabilityPolicyMining(log_path, output_dir)
    
    # Run full analysis with a subset of traces for demonstration
    # Use a smaller number for testing, or None for the full dataset
    framework.run_full_analysis(max_traces=200, violation_rate=0.05)
    
    print("Resource availability policy mining analysis with GT2 ground truth completed!")


if __name__ == "__main__":
    main()
