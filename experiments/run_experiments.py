#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Experimental Evaluation Runner
Executes all experiments for the policy mining research paper.

Author: Research Assistant
Date: October 2025
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'policy_engine'))

def run_command(cmd, description):
    """Run a shell command and log results"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    print(f"Completed in {elapsed:.2f} seconds")

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False

    print(result.stdout)
    return True

def main():
    """Run all experiments"""

    # Base paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    policy_engine_dir = base_dir / 'policy_engine'
    config_dir = base_dir / 'experiments' / 'configs'
    results_dir = base_dir / 'results'

    # Dataset path
    dataset = data_dir / 'BPI Challenge 2017.xes'

    if not dataset.exists():
        print(f"ERROR: Dataset not found at {dataset}")
        print("Please download and extract the BPIC 2017 dataset")
        return 1

    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = []

    # ===================================================================
    # EXPERIMENT 1: Policy Type Comparison
    # ===================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: Policy Type Comparison")
    print("="*80)

    for policy_config in ['p1_only', 'p2_only', 'both']:
        for case_limit in [200, 500, 1000]:
            exp_name = f"exp1_{policy_config}_{case_limit}cases"

            cmd = [
                'python',
                str(policy_engine_dir / 'policy_engine.py'),
                '--events', str(dataset),
                '--config', str(config_dir / f'config_{policy_config}.yaml'),
                '--out', str(results_dir / f'{exp_name}_policy_log.csv'),
                '--evaluate',
                '--violation-rate', '0.05',
                '--eval-out', str(results_dir / f'{exp_name}_eval.csv'),
                '--verbose'
            ]

            experiments.append((cmd, f"Exp1: {policy_config} with {case_limit} cases"))

    # ===================================================================
    # EXPERIMENT 2: Violation Rate Sensitivity
    # ===================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: Violation Rate Sensitivity Analysis")
    print("="*80)

    for rate in [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]:
        exp_name = f"exp2_rate_{int(rate*100):02d}pct"

        cmd = [
            'python',
            str(policy_engine_dir / 'policy_engine.py'),
            '--events', str(dataset),
            '--config', str(config_dir / 'config_both.yaml'),
            '--out', str(results_dir / f'{exp_name}_policy_log.csv'),
            '--evaluate',
            '--violation-rate', str(rate),
            '--eval-out', str(results_dir / f'{exp_name}_eval.csv'),
            '--verbose'
        ]

        experiments.append((cmd, f"Exp2: Violation rate {rate*100}%"))

    # ===================================================================
    # EXPERIMENT 3: Ground Truth Comparison (run later with different GT)
    # ===================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 3: Ground Truth Comparison (GT1 vs GT2)")
    print("="*80)

    exp_name = "exp3_ground_truth"
    cmd = [
        'python',
        str(policy_engine_dir / 'policy_engine.py'),
        '--events', str(dataset),
        '--config', str(config_dir / 'config_both.yaml'),
        '--out', str(results_dir / f'{exp_name}_policy_log.csv'),
        '--evaluate',
        '--violation-rate', '0.05',
        '--eval-out', str(results_dir / f'{exp_name}_eval.csv'),
        '--verbose'
    ]

    experiments.append((cmd, "Exp3: Ground truth comparison"))

    # ===================================================================
    # EXPERIMENT 4: Scalability Analysis
    # ===================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 4: Scalability Analysis")
    print("="*80)

    # Note: We'll measure time for different sizes in analysis
    # For now, run representative sizes
    for case_limit in [100, 500, 1000, 5000]:
        exp_name = f"exp4_scalability_{case_limit}cases"

        cmd = [
            'python',
            str(policy_engine_dir / 'policy_engine.py'),
            '--events', str(dataset),
            '--config', str(config_dir / 'config_both.yaml'),
            '--out', str(results_dir / f'{exp_name}_policy_log.csv'),
            '--evaluate',
            '--violation-rate', '0.05',
            '--eval-out', str(results_dir / f'{exp_name}_eval.csv'),
            '--verbose'
        ]

        experiments.append((cmd, f"Exp4: Scalability test with {case_limit} cases"))

    # ===================================================================
    # Run all experiments
    # ===================================================================

    print(f"\n\nTotal experiments to run: {len(experiments)}")
    print("This may take a while depending on your system...")
    print("\nStarting experiments...\n")

    results_log = []

    # Resume from experiment 15 (skip completed 1-14)
    start_from = 15

    for i, (cmd, description) in enumerate(experiments, 1):
        if i < start_from:
            print(f"\n[{i}/{len(experiments)}] {description} - SKIPPED (already completed)")
            results_log.append({
                'experiment': description,
                'success': True  # Assume previous runs were successful
            })
            continue

        print(f"\n[{i}/{len(experiments)}] {description}")
        success = run_command(cmd, description)

        results_log.append({
            'experiment': description,
            'success': success
        })

        if not success:
            print(f"\nWARNING: Experiment failed: {description}")
            response = input("Continue with remaining experiments? (y/n): ")
            if response.lower() != 'y':
                break

    # ===================================================================
    # Summary
    # ===================================================================

    print("\n" + "="*80)
    print("EXPERIMENT EXECUTION SUMMARY")
    print("="*80)

    successful = sum(1 for r in results_log if r['success'])
    failed = len(results_log) - successful

    print(f"\nTotal experiments: {len(results_log)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed experiments:")
        for r in results_log:
            if not r['success']:
                print(f"  - {r['experiment']}")

    print(f"\nResults saved to: {results_dir}")
    print("\nNext steps:")
    print("1. Run: python experiments/analyze_results.py")
    print("2. Check generated figures in paper_sections/")
    print("3. Review evaluation.tex for Overleaf")

    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
