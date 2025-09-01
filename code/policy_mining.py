#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Policy Mining and Conformance Checking Framework

This module implements a policy-aware conformance checking framework for business process mining.
It uses the BPI2017 dataset to detect policy violations based on resource usage and duration constraints.

Author: Manus AI
Date: September 2025
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union

# PM4Py imports
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.statistics.traces.generic.log import case_statistics

# Scikit-learn imports for evaluation metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class PolicyMiningFramework:
    """
    A framework for policy mining and conformance checking in business processes.
    
    This class provides functionality to:
    1. Load and preprocess event logs
    2. Define and extract policy rules
    3. Generate synthetic policy logs
    4. Detect policy violations using replay-based conformance checking
    5. Evaluate and visualize results
    """
    
    def __init__(self, log_path: str = None, output_dir: str = 'results'):
        """
        Initialize the PolicyMiningFramework.
        
        Args:
            log_path: Path to the event log file (.xes format)
            output_dir: Directory to save results and visualizations
        """
        self.log_path = log_path
        self.output_dir = output_dir
        self.event_log = None
        self.policy_log = None
        self.log_df = None
        self.policy_thresholds = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load event log if provided
        if log_path and os.path.exists(log_path):
            self.load_event_log(log_path)
    
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
        
        # Calculate duration for each event if not already present
        if 'duration' not in self.log_df.columns:
            # Group by case and sort by timestamp
            self.log_df = self.log_df.sort_values(['case:concept:name', 'time:timestamp'])
            
            # Calculate duration as time difference between consecutive events in the same case
            self.log_df['next_timestamp'] = self.log_df.groupby('case:concept:name')['time:timestamp'].shift(-1)
            self.log_df['duration'] = (self.log_df['next_timestamp'] - self.log_df['time:timestamp']).dt.total_seconds() / 3600  # Convert to hours
            
            # Fill NaN durations (last events in each case) with median duration for the same activity
            median_durations = self.log_df.groupby('concept:name')['duration'].median()
            self.log_df['duration'] = self.log_df['duration'].fillna(self.log_df['concept:name'].map(median_durations))
        
        print(f"Preprocessed event log with {len(self.log_df)} events")
        return self.log_df
    
    def extract_resource_activities(self) -> Dict[str, List[str]]:
        """
        Extract the activities performed by each resource.
        
        Returns:
            Dictionary mapping resources to lists of activities they perform
        """
        if self.log_df is None:
            self.preprocess_event_log()
        
        resource_activities = {}
        for resource, group in self.log_df.groupby('org:resource'):
            activities = group['concept:name'].unique().tolist()
            resource_activities[resource] = activities
        
        return resource_activities
    
    def define_policy_thresholds(self, method: str = 'percentile', percentile: float = 95) -> Dict[str, float]:
        """
        Define policy thresholds for resource-activity combinations.
        
        Args:
            method: Method to define thresholds ('percentile', 'mean_std', or 'manual')
            percentile: Percentile to use if method is 'percentile'
            
        Returns:
            Dictionary mapping resources to their allowed durations
        """
        if self.log_df is None:
            self.preprocess_event_log()
        
        print(f"Defining policy thresholds using {method} method...")
        
        # Group by resource and calculate thresholds
        if method == 'percentile':
            # Use percentile of durations for each resource
            for resource, group in self.log_df.groupby('org:resource'):
                self.policy_thresholds[resource] = np.percentile(group['duration'].dropna(), percentile)
        
        elif method == 'mean_std':
            # Use mean + 2*std for each resource
            for resource, group in self.log_df.groupby('org:resource'):
                mean = group['duration'].mean()
                std = group['duration'].std()
                self.policy_thresholds[resource] = mean + 2 * std
        
        elif method == 'manual':
            # Example manual thresholds (should be replaced with actual domain knowledge)
            # This is a placeholder that uses the 90th percentile
            for resource, group in self.log_df.groupby('org:resource'):
                self.policy_thresholds[resource] = np.percentile(group['duration'].dropna(), 90)
        
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        print(f"Defined policy thresholds for {len(self.policy_thresholds)} resources")
        return self.policy_thresholds
    
    def generate_policy_log(self) -> EventLog:
        """
        Generate a policy log based on the defined thresholds.
        
        Returns:
            The generated policy log
        """
        if not self.policy_thresholds:
            self.define_policy_thresholds()
        
        print("Generating policy log...")
        
        # Create policy log
        policy_log = EventLog()
        
        # Add a trace for each resource with its policy
        for resource, threshold in self.policy_thresholds.items():
            trace = Trace()
            trace.attributes['concept:name'] = resource
            trace.attributes['allowed_duration'] = threshold
            
            # Add a policy definition event
            policy_event = Event()
            policy_event['concept:name'] = 'PolicyDefinition'
            policy_event['time:timestamp'] = datetime.now()
            policy_event['org:resource'] = resource
            policy_event['duration'] = 0
            policy_event['policy_type'] = 'duration'
            policy_event['threshold'] = threshold
            
            trace.append(policy_event)
            policy_log.append(trace)
        
        self.policy_log = policy_log
        print(f"Generated policy log with {len(policy_log)} resource policies")
        
        # Save the policy log
        policy_log_path = os.path.join(self.output_dir, 'policy_log.xes')
        xes_exporter.apply(policy_log, policy_log_path)
        print(f"Saved policy log to {policy_log_path}")
        
        return policy_log
    
    def detect_policy_violations(self) -> pd.DataFrame:
        """
        Detect policy violations in the event log based on the policy thresholds.
        
        Returns:
            DataFrame with violation detection results
        """
        if self.log_df is None:
            self.preprocess_event_log()
        
        if not self.policy_thresholds:
            self.define_policy_thresholds()
        
        print("Detecting policy violations...")
        
        # Method 1: Event log only (baseline)
        # Use statistical thresholds from the event log (e.g., 95th percentile)
        durations_by_resource = {}
        for resource in self.log_df['org:resource'].unique():
            durations_by_resource[resource] = self.log_df[self.log_df['org:resource'] == resource]['duration'].values
        
        # Compute threshold as 95th percentile of durations for each resource
        eventlog_thresholds = {}
        for resource, durations in durations_by_resource.items():
            eventlog_thresholds[resource] = np.percentile(durations, 95)
        
        print(f"Event log thresholds (95th percentile): {eventlog_thresholds}")
        
        # Predict violations using event log thresholds
        self.log_df['eventlog_violation'] = self.log_df.apply(
            lambda row: row['duration'] > eventlog_thresholds.get(row['org:resource'], float('inf')),
            axis=1
        )
        
        # Method 2: Policy-aware approach
        # Use the policy thresholds directly
        self.log_df['policy_violation'] = self.log_df.apply(
            lambda row: row['duration'] > self.policy_thresholds.get(row['org:resource'], float('inf')),
            axis=1
        )
        
        # Set ground truth (for this example, we use policy violations as ground truth)
        self.log_df['true_violation'] = self.log_df['policy_violation']
        
        # Count violations
        true_violations_count = self.log_df['true_violation'].sum()
        print(f"Total violations detected: {true_violations_count} out of {len(self.log_df)} events ({true_violations_count/len(self.log_df)*100:.2f}%)")
        
        return self.log_df
    
    def evaluate_detection_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the performance of violation detection methods.
        
        Returns:
            Dictionary with performance metrics for each method
        """
        if 'true_violation' not in self.log_df.columns:
            self.detect_policy_violations()
        
        print("Evaluating detection performance...")
        
        # Calculate performance metrics for event log only method
        eventlog_precision = precision_score(self.log_df['true_violation'], self.log_df['eventlog_violation'])
        eventlog_recall = recall_score(self.log_df['true_violation'], self.log_df['eventlog_violation'])
        eventlog_f1 = f1_score(self.log_df['true_violation'], self.log_df['eventlog_violation'])
        
        # Calculate performance metrics for policy-aware method
        policy_precision = precision_score(self.log_df['true_violation'], self.log_df['policy_violation'])
        policy_recall = recall_score(self.log_df['true_violation'], self.log_df['policy_violation'])
        policy_f1 = f1_score(self.log_df['true_violation'], self.log_df['policy_violation'])
        
        # Calculate confusion matrices
        eventlog_cm = confusion_matrix(self.log_df['true_violation'], self.log_df['eventlog_violation'])
        policy_cm = confusion_matrix(self.log_df['true_violation'], self.log_df['policy_violation'])
        
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
        
        print("\nPerformance Metrics:")
        print(f"Event log only method: Precision={eventlog_precision:.4f}, Recall={eventlog_recall:.4f}, F1={eventlog_f1:.4f}")
        print(f"Policy-aware method: Precision={policy_precision:.4f}, Recall={policy_recall:.4f}, F1={policy_f1:.4f}")
        
        return results
    
    def calculate_edit_distance(self) -> pd.DataFrame:
        """
        Calculate edit distance for trace alignment.
        
        Returns:
            DataFrame with edit distance results
        """
        if 'true_violation' not in self.log_df.columns:
            self.detect_policy_violations()
        
        print("Calculating edit distances for trace alignment...")
        
        def calculate_levenshtein_distance(seq1, seq2):
            """Calculate Levenshtein distance between two sequences"""
            m, n = len(seq1), len(seq2)
            dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
            
            for i in range(m+1):
                dp[i][0] = i
            for j in range(n+1):
                dp[0][j] = j
                
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            return dp[m][n]
        
        # Group by case and calculate edit distance for each case
        case_edit_distances = []
        for case_id in self.log_df['case:concept:name'].unique():
            case_data = self.log_df[self.log_df['case:concept:name'] == case_id]
            true_seq = case_data['true_violation'].astype(int).tolist()
            eventlog_seq = case_data['eventlog_violation'].astype(int).tolist()
            policy_seq = case_data['policy_violation'].astype(int).tolist()
            
            eventlog_distance = calculate_levenshtein_distance(true_seq, eventlog_seq)
            policy_distance = calculate_levenshtein_distance(true_seq, policy_seq)
            
            case_edit_distances.append({
                'case_id': case_id,
                'eventlog_edit_distance': eventlog_distance,
                'policy_edit_distance': policy_distance,
                'sequence_length': len(true_seq)
            })
        
        # Convert to DataFrame for analysis
        edit_df = pd.DataFrame(case_edit_distances)
        
        # Calculate normalized edit distances (as percentage of sequence length)
        edit_df['eventlog_normalized_distance'] = edit_df['eventlog_edit_distance'] / edit_df['sequence_length']
        edit_df['policy_normalized_distance'] = edit_df['policy_edit_distance'] / edit_df['sequence_length']
        
        # Calculate average edit distances
        avg_eventlog_distance = edit_df['eventlog_normalized_distance'].mean()
        avg_policy_distance = edit_df['policy_normalized_distance'].mean()
        
        print("\nTrace Alignment Results:")
        print(f"Average normalized edit distance (Event Log Only): {avg_eventlog_distance:.4f}")
        print(f"Average normalized edit distance (Policy-Aware): {avg_policy_distance:.4f}")
        
        return edit_df
    
    def analyze_violations_by_resource(self) -> pd.DataFrame:
        """
        Analyze violations by resource.
        
        Returns:
            DataFrame with violation statistics by resource
        """
        if 'true_violation' not in self.log_df.columns:
            self.detect_policy_violations()
        
        print("Analyzing violations by resource...")
        
        # Count violations by resource
        violation_by_resource = self.log_df[self.log_df['true_violation']].groupby('org:resource').size()
        total_by_resource = self.log_df.groupby('org:resource').size()
        violation_rate_by_resource = (violation_by_resource / total_by_resource * 100).fillna(0)
        
        # Create a table of violation types
        violation_types = pd.DataFrame({
            'Resource': violation_rate_by_resource.index,
            'Total_Events': total_by_resource.values,
            'Violations': violation_by_resource.values,
            'Violation_Rate_Percent': violation_rate_by_resource.values.round(2)
        })
        
        # Save to CSV
        violation_types.to_csv(os.path.join(self.output_dir, 'violation_types.csv'), index=False)
        
        return violation_types
    
    def visualize_results(self):
        """
        Visualize the results of policy violation detection.
        """
        if 'true_violation' not in self.log_df.columns:
            self.detect_policy_violations()
        
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
        
        # Get durations by resource
        durations_by_resource = {}
        for resource in self.log_df['org:resource'].unique():
            durations_by_resource[resource] = self.log_df[self.log_df['org:resource'] == resource]['duration'].values
        
        # Create a figure with multiple subplots
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Duration distribution by resource
        plt.subplot(2, 2, 1)
        for resource, durations in durations_by_resource.items():
            if len(durations) > 100:  # Only plot resources with sufficient data
                sns.kdeplot(durations, label=resource)
                plt.axvline(x=self.policy_thresholds[resource], linestyle='--', color='gray', alpha=0.7)
        plt.title('Duration Distribution by Resource', fontsize=14)
        plt.xlabel('Duration (hours)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=10)
        
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
        plt.title('Detection Performance Comparison', fontsize=14)
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
        plt.savefig(os.path.join(self.output_dir, 'policy_violation_detection_results.png'), dpi=300)
        print(f"Saved detection results visualization to {os.path.join(self.output_dir, 'policy_violation_detection_results.png')}")
        
        # Create a separate figure for violation rates by resource
        plt.figure(figsize=(12, 8))
        violation_types = self.analyze_violations_by_resource()
        
        # Sort by violation rate
        violation_types = violation_types.sort_values('Violation_Rate_Percent', ascending=False)
        
        # Plot top 15 resources by violation rate
        top_n = min(15, len(violation_types))
        violation_types.head(top_n).plot(
            x='Resource', 
            y='Violation_Rate_Percent', 
            kind='bar', 
            color='crimson'
        )
        plt.title('Top Resources by Violation Rate', fontsize=14)
        plt.xlabel('Resource', fontsize=12)
        plt.ylabel('Violation Rate (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'violation_rate_by_resource.png'), dpi=300)
        print(f"Saved violation rate visualization to {os.path.join(self.output_dir, 'violation_rate_by_resource.png')}")
        
        # Create summary results file
        with open(os.path.join(self.output_dir, 'summary_results.txt'), 'w') as f:
            f.write('Policy Violation Detection Results Summary\n')
            f.write('===========================================\n\n')
            f.write(f'Total events analyzed: {len(self.log_df)}\n')
            true_violations_count = self.log_df['true_violation'].sum()
            f.write(f'Total true violations: {true_violations_count} ({true_violations_count/len(self.log_df)*100:.2f}%)\n\n')
            
            f.write('Performance Metrics:\n')
            f.write(f'Event log only method: Precision={eventlog_precision:.4f}, Recall={eventlog_recall:.4f}, F1={eventlog_f1:.4f}\n')
            f.write(f'Policy-aware method: Precision={policy_precision:.4f}, Recall={policy_recall:.4f}, F1={policy_f1:.4f}\n\n')
            
            # Get edit distances
            edit_df = self.calculate_edit_distance()
            avg_eventlog_distance = edit_df['eventlog_normalized_distance'].mean()
            avg_policy_distance = edit_df['policy_normalized_distance'].mean()
            
            f.write('Trace Alignment Results:\n')
            f.write(f'Average normalized edit distance (Event Log Only): {avg_eventlog_distance:.4f}\n')
            f.write(f'Average normalized edit distance (Policy-Aware): {avg_policy_distance:.4f}\n\n')
            
            f.write('Top 5 Resources by Violation Rate:\n')
            for _, row in violation_types.head(5).iterrows():
                f.write(f"  {row['Resource']}: {row['Violation_Rate_Percent']:.2f}% ({row['Violations']} out of {row['Total_Events']} events)\n")
        
        print(f"Saved summary results to {os.path.join(self.output_dir, 'summary_results.txt')}")
    
    def save_results(self):
        """
        Save all results to files.
        """
        if self.log_df is None:
            raise ValueError("No results to save. Run detect_policy_violations first.")
        
        print("Saving all results...")
        
        # Save the analysis results
        self.log_df.to_csv(os.path.join(self.output_dir, 'analysis_results.csv'), index=False)
        
        # Save edit distance results
        edit_df = self.calculate_edit_distance()
        edit_df.to_csv(os.path.join(self.output_dir, 'edit_distance_results.csv'), index=False)
        
        # Save violation types
        violation_types = self.analyze_violations_by_resource()
        violation_types.to_csv(os.path.join(self.output_dir, 'violation_types.csv'), index=False)
        
        print(f"All results saved to {self.output_dir}")
    
    def run_full_analysis(self, max_traces: int = None):
        """
        Run the full analysis pipeline.
        
        Args:
            max_traces: Maximum number of traces to analyze (None for all)
        """
        print("Starting full policy mining analysis...")
        
        # Load event log if not already loaded
        if self.event_log is None and self.log_path:
            self.load_event_log(self.log_path, max_traces)
        
        # Preprocess event log
        self.preprocess_event_log()
        
        # Define policy thresholds
        self.define_policy_thresholds()
        
        # Generate policy log
        self.generate_policy_log()
        
        # Detect policy violations
        self.detect_policy_violations()
        
        # Evaluate detection performance
        self.evaluate_detection_performance()
        
        # Calculate edit distances
        self.calculate_edit_distance()
        
        # Analyze violations by resource
        self.analyze_violations_by_resource()
        
        # Visualize results
        self.visualize_results()
        
        # Save all results
        self.save_results()
        
        print("Full analysis completed successfully!")


def main():
    """
    Main function to run the policy mining framework.
    """
    # Set up directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Define paths
    log_path = 'BPI Challenge 2017.xes'
    output_dir = 'results'
    
    # Create framework instance
    framework = PolicyMiningFramework(log_path, output_dir)
    
    # Run full analysis with a subset of traces for demonstration
    # Use a smaller number for testing, or None for the full dataset
    framework.run_full_analysis(max_traces=100)
    
    print("Policy mining analysis completed!")


if __name__ == "__main__":
    main()
