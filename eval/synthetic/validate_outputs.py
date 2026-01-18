#!/usr/bin/env python3
"""
Sanity-check validation for synthetic evaluation outputs.

Verifies:
- ground_truth.csv and predictions.csv join 1:1 on case_id
- PR curve computed from continuous scores (severity), not binary predictions
- Counts match (n_cases, n_violations_gt, n_predicted)
- All required files exist
- Data integrity (no NaN, proper types)

Prints PASS/FAIL report.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

from config import SyntheticConfig


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def validate_files_exist(config: SyntheticConfig) -> List[str]:
    """Check that all required output files exist."""
    required_files = [
        'event_log.csv',
        'event_log.xes',
        'ground_truth.csv',
        'predictions.csv',
        'pr_curve.csv',
        'metrics.json'
    ]

    errors = []
    for filename in required_files:
        path = config.get_output_path(filename)
        if not path.exists():
            errors.append(f"Missing required file: {filename}")

    return errors


def validate_ground_truth_predictions_join(config: SyntheticConfig) -> List[str]:
    """Validate that ground_truth and predictions join 1:1 on case_id."""
    errors = []

    gt = pd.read_csv(config.get_output_path('ground_truth.csv'))
    pred = pd.read_csv(config.get_output_path('predictions.csv'))

    # Check for duplicates
    if gt['case_id'].duplicated().any():
        errors.append("ground_truth.csv has duplicate case_ids")

    if pred['case_id'].duplicated().any():
        errors.append("predictions.csv has duplicate case_ids")

    # Check for 1:1 join
    gt_ids = set(gt['case_id'])
    pred_ids = set(pred['case_id'])

    missing_in_pred = gt_ids - pred_ids
    missing_in_gt = pred_ids - gt_ids

    if missing_in_pred:
        errors.append(f"predictions.csv missing {len(missing_in_pred)} case_ids from ground_truth")

    if missing_in_gt:
        errors.append(f"ground_truth.csv missing {len(missing_in_gt)} case_ids from predictions")

    # Check counts match
    if len(gt) != len(pred):
        errors.append(f"Row count mismatch: ground_truth={len(gt)}, predictions={len(pred)}")

    return errors


def validate_pr_curve_from_scores(config: SyntheticConfig) -> List[str]:
    """Validate that PR curve was computed from continuous scores, not binary predictions."""
    errors = []

    pred = pd.read_csv(config.get_output_path('predictions.csv'))
    pr_curve = pd.read_csv(config.get_output_path('pr_curve.csv'))

    # Check that predictions has severity scores
    if 'severity' not in pred.columns:
        errors.append("predictions.csv missing 'severity' column (continuous scores required for PR curve)")
        return errors

    # Check severity is continuous (not just binary 0/1)
    unique_severity = pred['severity'].nunique()
    if unique_severity < 10:
        errors.append(f"severity appears binary/discrete ({unique_severity} unique values), expected continuous scores")

    # Check severity range is [0, 1]
    severity_min = pred['severity'].min()
    severity_max = pred['severity'].max()

    if severity_min < -0.01 or severity_max > 1.01:
        errors.append(f"severity scores out of expected [0, 1] range: [{severity_min:.3f}, {severity_max:.3f}]")

    # Check PR curve has expected structure
    required_pr_cols = ['threshold', 'precision', 'recall']
    missing_cols = [col for col in required_pr_cols if col not in pr_curve.columns]
    if missing_cols:
        errors.append(f"pr_curve.csv missing columns: {missing_cols}")

    # Check PR curve has multiple points
    if len(pr_curve) < 5:
        errors.append(f"pr_curve.csv has only {len(pr_curve)} points, expected many points from continuous scores")

    return errors


def validate_counts_match(config: SyntheticConfig) -> List[str]:
    """Validate that counts are consistent across files."""
    errors = []

    event_log = pd.read_csv(config.get_output_path('event_log.csv'))
    gt = pd.read_csv(config.get_output_path('ground_truth.csv'))
    pred = pd.read_csv(config.get_output_path('predictions.csv'))

    # Number of cases
    n_cases_log = event_log['case_id'].nunique()
    n_cases_gt = len(gt)
    n_cases_pred = len(pred)

    if n_cases_log != n_cases_gt:
        errors.append(f"Case count mismatch: event_log={n_cases_log}, ground_truth={n_cases_gt}")

    if n_cases_log != n_cases_pred:
        errors.append(f"Case count mismatch: event_log={n_cases_log}, predictions={n_cases_pred}")

    # Number of violations
    n_violations_gt = gt['is_violation'].sum()
    n_violations_pred = pred['is_violation'].sum()

    # NOTE: Some mismatch is EXPECTED due to:
    # - False positives from noise (predictions > ground truth)
    # - False negatives (predictions < ground truth)
    # Only flag if EXTREMELY different (>200% or predictions < 50% of ground truth)

    violation_ratio = n_violations_pred / max(n_violations_gt, 1)

    # Warnings only - don't fail on this
    if violation_ratio < 0.5:
        errors.append(f"Very low recall: predicted {n_violations_pred} violations vs {n_violations_gt} ground truth ({100*violation_ratio:.1f}%)")
    elif violation_ratio > 3.0:
        errors.append(f"Extremely high false positive rate: predicted {n_violations_pred} violations vs {n_violations_gt} ground truth ({100*violation_ratio:.1f}%)")

    return errors


def validate_data_integrity(config: SyntheticConfig) -> List[str]:
    """Validate data integrity (no NaN in critical columns, proper types)."""
    errors = []

    gt = pd.read_csv(config.get_output_path('ground_truth.csv'))
    pred = pd.read_csv(config.get_output_path('predictions.csv'))

    # Check for NaN in critical columns
    critical_gt_cols = ['case_id', 'is_violation', 'outcome']
    for col in critical_gt_cols:
        if col in gt.columns and gt[col].isna().any():
            errors.append(f"ground_truth.csv has NaN values in column '{col}'")

    critical_pred_cols = ['case_id', 'is_violation', 'severity', 'outcome']
    for col in critical_pred_cols:
        if col in pred.columns and pred[col].isna().any():
            errors.append(f"predictions.csv has NaN values in column '{col}'")

    # Check boolean columns are actually boolean/binary
    if 'is_violation' in gt.columns:
        unique_vals = set(gt['is_violation'].dropna().unique())
        if not unique_vals.issubset({0, 1, True, False}):
            errors.append(f"ground_truth.is_violation has non-boolean values: {unique_vals}")

    if 'is_violation' in pred.columns:
        unique_vals = set(pred['is_violation'].dropna().unique())
        if not unique_vals.issubset({0, 1, True, False}):
            errors.append(f"predictions.is_violation has non-boolean values: {unique_vals}")

    return errors


def validate_metrics_json(config: SyntheticConfig) -> List[str]:
    """Validate metrics.json structure and values."""
    errors = []

    import json

    metrics_path = config.get_output_path('metrics.json')
    if not metrics_path.exists():
        return ["metrics.json file missing"]

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Check for required metrics
    if 'pr_curve_metrics' not in metrics:
        errors.append("metrics.json missing 'pr_curve_metrics' section")
    else:
        pr_metrics = metrics['pr_curve_metrics']

        required_pr_metrics = ['average_precision', 'brier_score']
        for metric in required_pr_metrics:
            if metric not in pr_metrics:
                errors.append(f"metrics.json missing PR metric: {metric}")
            else:
                value = pr_metrics[metric]
                # Check reasonable range [0, 1]
                if not (0 <= value <= 1):
                    errors.append(f"metrics.json {metric} out of range [0, 1]: {value}")

    return errors


def run_validation(run_id: str) -> Tuple[bool, List[str], List[str]]:
    """Run all validation checks and return (success, errors, warnings)."""

    config = SyntheticConfig(run_id=run_id)

    all_errors = []
    warnings = []

    print(f"Validating run: {run_id}")
    print("="*80)

    # 1. Files exist
    print("[1/6] Checking required files exist...")
    errors = validate_files_exist(config)
    all_errors.extend(errors)
    if errors:
        print(f"  FAIL: {len(errors)} missing files")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  PASS")

    # If critical files missing, can't continue
    if any('event_log' in e or 'ground_truth' in e or 'predictions' in e for e in all_errors):
        return False, all_errors, warnings

    # 2. Ground truth and predictions join
    print("[2/6] Checking ground_truth <-> predictions 1:1 join...")
    errors = validate_ground_truth_predictions_join(config)
    all_errors.extend(errors)
    if errors:
        print(f"  FAIL: {len(errors)} join issues")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  PASS")

    # 3. PR curve from scores
    print("[3/6] Checking PR curve from continuous scores...")
    errors = validate_pr_curve_from_scores(config)
    all_errors.extend(errors)
    if errors:
        print(f"  FAIL: {len(errors)} PR curve issues")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  PASS")

    # 4. Counts match
    print("[4/6] Checking counts match across files...")
    errors = validate_counts_match(config)
    all_errors.extend(errors)
    if errors:
        print(f"  FAIL: {len(errors)} count mismatches")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  PASS")

    # 5. Data integrity
    print("[5/6] Checking data integrity...")
    errors = validate_data_integrity(config)
    all_errors.extend(errors)
    if errors:
        print(f"  FAIL: {len(errors)} integrity issues")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  PASS")

    # 6. Metrics JSON
    print("[6/6] Checking metrics.json structure...")
    errors = validate_metrics_json(config)
    all_errors.extend(errors)
    if errors:
        print(f"  FAIL: {len(errors)} metrics issues")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  PASS")

    success = len(all_errors) == 0

    return success, all_errors, warnings


def print_summary(success: bool, errors: List[str], warnings: List[str]):
    """Print final validation summary."""
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    if success:
        print("\n[PASS] ALL CHECKS PASSED")
        print("\nOutputs are valid and ready for paper submission.")
    else:
        print(f"\n[FAIL] VALIDATION FAILED ({len(errors)} errors)")
        print("\nErrors found:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")

    if warnings:
        print(f"\n[WARN] {len(warnings)} warnings:")
        for i, warn in enumerate(warnings, 1):
            print(f"  {i}. {warn}")

    print()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        # Use latest run
        runs = sorted([r for r in Path('eval/synthetic/outputs').glob('*/')
                       if r.name != 'multiseed'])
        if not runs:
            print("ERROR: No run outputs found.")
            sys.exit(1)

        run_id = runs[-1].name
        print(f"No run_id specified, using latest: {run_id}")

    success, errors, warnings = run_validation(run_id)

    print_summary(success, errors, warnings)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
