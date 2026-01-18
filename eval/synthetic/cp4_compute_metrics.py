#!/usr/bin/env python3
"""
CP4: Compute metrics (precision/recall/F1) and generate plots.

Compares ground truth labels with policy checker predictions to evaluate
the correctness of the policy-only conformance checking approach.

Outputs:
- metrics.json (detailed metrics)
- metrics.csv (summary table)
- confusion_matrix.png
- metrics_by_outcome.png
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from config import SyntheticConfig


def load_data(config: SyntheticConfig) -> tuple:
    """Load ground truth and predictions."""
    gt_path = config.get_output_path('ground_truth.csv')
    pred_path = config.get_output_path('predictions.csv')

    print(f"Loading ground truth from: {gt_path}")
    ground_truth = pd.read_csv(gt_path)

    print(f"Loading predictions from: {pred_path}")
    predictions = pd.read_csv(pred_path)

    return ground_truth, predictions


def compute_binary_metrics(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
    """
    Compute binary classification metrics (violation vs non-violation).

    This is the primary metric for policy-only conformance checking:
    - Positive class: duty_unmet (violation)
    - Negative class: all others (conforming or not applicable)
    """
    # Merge datasets
    merged = ground_truth.merge(predictions, on='case_id', suffixes=('_gt', '_pred'))

    # Binary labels: violation (1) vs non-violation (0)
    y_true = (merged['outcome_gt'] == 'duty_unmet').astype(int)
    y_pred = (merged['outcome_pred'] == 'duty_unmet').astype(int)

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'binary_classification': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'support': {
                'violations': int(y_true.sum()),
                'non_violations': int((1 - y_true).sum()),
                'total': int(len(y_true))
            }
        }
    }

    return metrics, merged


def compute_multiclass_metrics(merged: pd.DataFrame) -> Dict:
    """
    Compute multiclass classification metrics (all four outcomes).

    Outcomes:
    - not_applicable
    - duty_met
    - duty_met_via_delegation
    - duty_unmet
    """
    y_true = merged['outcome_gt']
    y_pred = merged['outcome_pred']

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Confusion matrix
    labels = ['not_applicable', 'duty_met', 'duty_met_via_delegation', 'duty_unmet']
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        'multiclass_classification': {
            'accuracy': float(accuracy),
            'per_class_metrics': {
                label: {
                    'precision': float(report[label]['precision']),
                    'recall': float(report[label]['recall']),
                    'f1_score': float(report[label]['f1-score']),
                    'support': int(report[label]['support'])
                }
                for label in labels if label in report
            },
            'macro_avg': {
                'precision': float(report['macro avg']['precision']),
                'recall': float(report['macro avg']['recall']),
                'f1_score': float(report['macro avg']['f1-score'])
            },
            'weighted_avg': {
                'precision': float(report['weighted avg']['precision']),
                'recall': float(report['weighted avg']['recall']),
                'f1_score': float(report['weighted avg']['f1-score'])
            },
            'confusion_matrix': cm.tolist()
        }
    }

    return metrics


def compute_applicable_only_metrics(merged: pd.DataFrame) -> Dict:
    """
    Compute metrics for applicable cases only (requires_senior == True).

    This focuses on cases where the policy actually applies.
    """
    applicable = merged[merged['requires_senior_gt'] == True]

    if len(applicable) == 0:
        return {'applicable_only': {'note': 'No applicable cases'}}

    y_true = applicable['outcome_gt']
    y_pred = applicable['outcome_pred']

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Binary: violation vs conforming (within applicable cases)
    y_true_binary = (y_true == 'duty_unmet').astype(int)
    y_pred_binary = (y_pred == 'duty_unmet').astype(int)

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    metrics = {
        'applicable_only': {
            'total_applicable_cases': int(len(applicable)),
            'accuracy': float(accuracy),
            'violation_detection': {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'true_violations': int(y_true_binary.sum()),
                'predicted_violations': int(y_pred_binary.sum())
            }
        }
    }

    return metrics


def save_metrics(all_metrics: Dict, config: SyntheticConfig):
    """Save metrics to JSON and CSV."""
    # Save JSON
    json_path = config.get_output_path('metrics.json')
    print(f"Saving metrics to: {json_path}")
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Create CSV summary
    csv_data = []

    # Binary classification row
    binary = all_metrics['binary_classification']
    csv_data.append({
        'metric_type': 'binary_violation_detection',
        'precision': binary['precision'],
        'recall': binary['recall'],
        'f1_score': binary['f1_score'],
        'accuracy': binary['accuracy'],
        'support': binary['support']['violations']
    })

    # Multiclass overall
    multi = all_metrics['multiclass_classification']
    csv_data.append({
        'metric_type': 'multiclass_macro_avg',
        'precision': multi['macro_avg']['precision'],
        'recall': multi['macro_avg']['recall'],
        'f1_score': multi['macro_avg']['f1_score'],
        'accuracy': multi['accuracy'],
        'support': all_metrics['binary_classification']['support']['total']
    })

    # Applicable only
    if 'applicable_only' in all_metrics and 'accuracy' in all_metrics['applicable_only']:
        appl = all_metrics['applicable_only']['violation_detection']
        csv_data.append({
            'metric_type': 'applicable_only_violations',
            'precision': appl['precision'],
            'recall': appl['recall'],
            'f1_score': appl['f1_score'],
            'accuracy': all_metrics['applicable_only']['accuracy'],
            'support': appl['true_violations']
        })

    csv_df = pd.DataFrame(csv_data)
    csv_path = config.get_output_path('metrics.csv')
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved summary to: {csv_path}")


def create_confusion_matrix_plot(merged: pd.DataFrame, config: SyntheticConfig):
    """Create confusion matrix heatmap for multiclass classification."""
    y_true = merged['outcome_gt']
    y_pred = merged['outcome_pred']

    labels = ['not_applicable', 'duty_met', 'duty_met_via_delegation', 'duty_unmet']
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'},
                ax=ax, linewidths=1, linecolor='black')

    ax.set_xlabel('Predicted Outcome', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Outcome', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix: Policy-Only Conformance Checking',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    output_path = config.get_output_path('confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to: {output_path}")
    plt.close()


def create_metrics_bar_chart(all_metrics: Dict, config: SyntheticConfig):
    """Create bar chart comparing metrics across different evaluation scopes."""
    # Extract data
    data = []

    # Binary classification
    binary = all_metrics['binary_classification']
    data.append({
        'Scope': 'Binary\n(Violation Detection)',
        'Precision': binary['precision'],
        'Recall': binary['recall'],
        'F1-Score': binary['f1_score']
    })

    # Multiclass macro avg
    multi = all_metrics['multiclass_classification']['macro_avg']
    data.append({
        'Scope': 'Multiclass\n(Macro Avg)',
        'Precision': multi['precision'],
        'Recall': multi['recall'],
        'F1-Score': multi['f1_score']
    })

    # Applicable only
    if 'applicable_only' in all_metrics and 'accuracy' in all_metrics['applicable_only']:
        appl = all_metrics['applicable_only']['violation_detection']
        data.append({
            'Scope': 'Applicable Only\n(Violations)',
            'Precision': appl['precision'],
            'Recall': appl['recall'],
            'F1-Score': appl['f1_score']
        })

    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.25

    bars1 = ax.bar(x - width, df['Precision'], width, label='Precision',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, df['Recall'], width, label='Recall',
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, df['F1-Score'], width, label='F1-Score',
                   color='#e74c3c', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Evaluation Scope', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Policy-Only Conformance Checking: Metrics by Scope',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Scope'])
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()

    output_path = config.get_output_path('metrics_by_scope.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics bar chart to: {output_path}")
    plt.close()


def print_summary(all_metrics: Dict):
    """Print summary to console."""
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)

    # Binary classification
    binary = all_metrics['binary_classification']
    print("\nBinary Classification (Violation Detection):")
    print(f"  Precision: {binary['precision']:.3f}")
    print(f"  Recall:    {binary['recall']:.3f}")
    print(f"  F1-Score:  {binary['f1_score']:.3f}")
    print(f"  Accuracy:  {binary['accuracy']:.3f}")
    print(f"  Support:   {binary['support']['violations']} violations / "
          f"{binary['support']['total']} total cases")

    # Confusion matrix
    cm = binary['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives:  {cm['true_positives']}")
    print(f"    False Positives: {cm['false_positives']}")
    print(f"    True Negatives:  {cm['true_negatives']}")
    print(f"    False Negatives: {cm['false_negatives']}")

    # Multiclass
    multi = all_metrics['multiclass_classification']
    print("\nMulticlass Classification (All Outcomes):")
    print(f"  Overall Accuracy: {multi['accuracy']:.3f}")
    print(f"  Macro Avg F1:     {multi['macro_avg']['f1_score']:.3f}")

    print("\n  Per-Class Metrics:")
    for label, metrics in multi['per_class_metrics'].items():
        print(f"    {label:30s}: P={metrics['precision']:.3f} "
              f"R={metrics['recall']:.3f} F1={metrics['f1_score']:.3f} "
              f"(n={metrics['support']})")

    # Applicable only
    if 'applicable_only' in all_metrics and 'accuracy' in all_metrics['applicable_only']:
        appl = all_metrics['applicable_only']
        print("\nApplicable Cases Only (requires_senior=True):")
        print(f"  Total applicable: {appl['total_applicable_cases']}")
        print(f"  Accuracy:         {appl['accuracy']:.3f}")
        print(f"  Violation Detection:")
        vd = appl['violation_detection']
        print(f"    Precision: {vd['precision']:.3f}")
        print(f"    Recall:    {vd['recall']:.3f}")
        print(f"    F1-Score:  {vd['f1_score']:.3f}")


def main():
    """Main execution for CP4."""
    print("="*80)
    print("CP4: COMPUTE METRICS AND GENERATE PLOTS")
    print("="*80)

    # Load configuration
    runs = sorted(Path('eval/synthetic/outputs').glob('*/'))
    if not runs:
        print("ERROR: No outputs found. Run CP1-CP3 first.")
        return

    latest_run = runs[-1]
    run_id = latest_run.name

    config = SyntheticConfig(run_id=run_id)
    print(f"\nRun ID: {config.run_id}")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    ground_truth, predictions = load_data(config)
    print(f"Loaded {len(ground_truth)} ground truth labels")
    print(f"Loaded {len(predictions)} predictions")

    # Compute binary metrics
    print("\n" + "="*80)
    print("COMPUTING BINARY METRICS")
    print("="*80)
    binary_metrics, merged = compute_binary_metrics(ground_truth, predictions)

    # Compute multiclass metrics
    print("\n" + "="*80)
    print("COMPUTING MULTICLASS METRICS")
    print("="*80)
    multiclass_metrics = compute_multiclass_metrics(merged)

    # Compute applicable-only metrics
    print("\n" + "="*80)
    print("COMPUTING APPLICABLE-ONLY METRICS")
    print("="*80)
    applicable_metrics = compute_applicable_only_metrics(merged)

    # Combine all metrics
    all_metrics = {**binary_metrics, **multiclass_metrics, **applicable_metrics}

    # Print summary
    print_summary(all_metrics)

    # Save metrics
    print("\n" + "="*80)
    print("SAVING METRICS")
    print("="*80)
    save_metrics(all_metrics, config)

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    create_confusion_matrix_plot(merged, config)
    create_metrics_bar_chart(all_metrics, config)

    print("\n" + "="*80)
    print("CP4 COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {config.get_output_path('metrics.json')}")
    print(f"  - {config.get_output_path('metrics.csv')}")
    print(f"  - {config.get_output_path('confusion_matrix.png')}")
    print(f"  - {config.get_output_path('metrics_by_scope.png')}")


if __name__ == '__main__':
    main()
