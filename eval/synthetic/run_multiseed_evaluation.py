#!/usr/bin/env python3
"""
Checkpoint 3: Multi-seed evaluation with statistics.

Runs the complete pipeline (CP1 → CP2 noisy → CP3 enhanced) multiple times
with different seeds to compute mean/std for all metrics.

Outputs:
- seeds_summary.csv: Summary statistics across all seeds
- seeds_details.csv: Per-seed detailed results
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np

from config import SyntheticConfig


def run_single_seed(seed: int, output_base: Path) -> Dict:
    """
    Run complete pipeline for a single seed.

    Returns dict with all metrics for this seed.
    """
    print(f"\n{'='*80}")
    print(f"RUNNING SEED {seed}")
    print(f"{'='*80}\n")

    # Create config with this seed
    config = SyntheticConfig(seed=seed)
    run_id = config.run_id

    print(f"Run ID: {run_id}")

    # Import and run each stage programmatically to avoid subprocess overhead
    try:
        # CP1: Generate clean log
        print("\n[1/3] Generating clean log...")
        from cp1_generate_clean_log import generate_simple_process_log, export_log_to_xes, export_log_to_csv
        config.create_output_dir()
        log = generate_simple_process_log(config)
        export_log_to_xes(log, config.get_output_path('event_log.xes'))
        export_log_to_csv(log, config.get_output_path('event_log.csv'))

        # CP2: Inject violations with noise
        print("\n[2/3] Injecting violations with noise...")
        from cp2_inject_violations_noisy import NoisyViolationInjector
        from pm4py.objects.log.importer.xes import importer as xes_importer

        clean_log = xes_importer.apply(str(config.get_output_path('event_log.xes')))
        injector = NoisyViolationInjector(config)
        modified_log, ground_truth_df = injector.inject_violations(clean_log)

        # Export modified log
        from pm4py.objects.log.exporter.xes import exporter as xes_exporter
        xes_exporter.apply(modified_log, str(config.get_output_path('event_log.xes')))

        # Save to CSV
        rows = []
        for trace in modified_log:
            case_id = trace.attributes['concept:name']
            amount = trace.attributes.get('amount', 0)
            for event in trace:
                rows.append({
                    'case_id': case_id,
                    'activity': event['concept:name'],
                    'timestamp': event['time:timestamp'],
                    'resource': event.get('org:resource', ''),
                    'role': event.get('role', ''),
                    'amount': amount
                })
        log_df = pd.DataFrame(rows)
        log_df.to_csv(config.get_output_path('event_log.csv'), index=False)
        ground_truth_df.to_csv(config.get_output_path('ground_truth.csv'), index=False)

        # CP3: Run enhanced policy checker
        print("\n[3/3] Running enhanced policy checker...")
        from cp3_enhanced_policy_checker import (
            load_synthetic_event_log, run_enhanced_policy_checker,
            compute_pr_curve_metrics
        )

        df = load_synthetic_event_log(config.get_output_path('event_log.csv'))
        predictions_df = run_enhanced_policy_checker(
            df, config.senior_approval_threshold,
            config.delegation_window_hours, config
        )
        pr_df, pr_metrics = compute_pr_curve_metrics(ground_truth_df, predictions_df, config)

        # Save outputs
        predictions_df.to_csv(config.get_output_path('predictions.csv'), index=False)
        pr_df.to_csv(config.get_output_path('pr_curve.csv'), index=False)

        with open(config.get_output_path('metrics.json'), 'w') as f:
            json.dump({'pr_curve_metrics': pr_metrics}, f, indent=2)

        # Compute binary classification metrics
        merged = ground_truth_df.merge(predictions_df, on='case_id', suffixes=('_gt', '_pred'))
        y_true = merged['is_violation_gt'].astype(int)
        y_pred = merged['is_violation_pred'].astype(int)

        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        # Collect metrics
        metrics = {
            'seed': seed,
            'run_id': run_id,
            'num_cases': len(ground_truth_df),
            'num_violations_gt': int(y_true.sum()),
            'num_violations_pred': int(y_pred.sum()),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'average_precision': float(pr_metrics['average_precision']),
            'brier_score': float(pr_metrics['brier_score']),
            'noise_near_miss': injector.noise_stats['near_miss_cases'],
            'noise_multiple_approval': injector.noise_stats['multiple_approval_cases'],
            'noise_missing_role': injector.noise_stats['missing_role_events'],
        }

        print(f"\nSeed {seed} complete:")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print(f"  PR-AUC: {pr_metrics['average_precision']:.3f}")

        return metrics

    except Exception as e:
        print(f"ERROR in seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_multiseed_evaluation(seeds: List[int], output_dir: Path):
    """Run evaluation across multiple seeds and aggregate results."""
    print("="*80)
    print("MULTI-SEED EVALUATION (Checkpoint 3)")
    print("="*80)
    print(f"\nNumber of seeds: {len(seeds)}")
    print(f"Seeds: {seeds}")

    all_results = []

    for seed in seeds:
        result = run_single_seed(seed, output_dir)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("ERROR: No successful runs!")
        return

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save detailed results
    details_path = output_dir / 'seeds_details.csv'
    results_df.to_csv(details_path, index=False)
    print(f"\nSaved detailed results to: {details_path}")

    # Compute summary statistics
    metric_cols = ['precision', 'recall', 'f1_score', 'accuracy',
                   'average_precision', 'brier_score']

    summary_data = []
    for col in metric_cols:
        summary_data.append({
            'metric': col,
            'mean': results_df[col].mean(),
            'std': results_df[col].std(),
            'min': results_df[col].min(),
            'max': results_df[col].max(),
            'median': results_df[col].median()
        })

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_path = output_dir / 'seeds_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary statistics to: {summary_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY STATISTICS ACROSS SEEDS")
    print("="*80)
    print(f"\nNumber of runs: {len(all_results)}")
    print("\nMetric Statistics:")
    print(summary_df.to_string(index=False))

    # Additional statistics
    print(f"\nViolations (ground truth):")
    print(f"  Mean: {results_df['num_violations_gt'].mean():.1f} ± {results_df['num_violations_gt'].std():.1f}")
    print(f"  Range: [{results_df['num_violations_gt'].min()}, {results_df['num_violations_gt'].max()}]")

    print(f"\nViolations (predicted):")
    print(f"  Mean: {results_df['num_violations_pred'].mean():.1f} ± {results_df['num_violations_pred'].std():.1f}")
    print(f"  Range: [{results_df['num_violations_pred'].min()}, {results_df['num_violations_pred'].max()}]")


def main():
    """Main execution for multi-seed evaluation."""
    # Use seeds 42-51 (10 runs)
    seeds = list(range(42, 52))

    output_dir = Path('eval/synthetic/outputs/multiseed')
    output_dir.mkdir(parents=True, exist_ok=True)

    run_multiseed_evaluation(seeds, output_dir)

    print("\n" + "="*80)
    print("MULTI-SEED EVALUATION COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'seeds_summary.csv'}")
    print(f"  - {output_dir / 'seeds_details.csv'}")
    print(f"\nIndividual run outputs in: eval/synthetic/outputs/<run_id>/")


if __name__ == '__main__':
    main()
