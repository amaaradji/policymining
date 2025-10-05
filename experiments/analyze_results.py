#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Results Analysis and Visualization for Policy Mining Experiments

Aggregates all experiment results and generates publication-ready figures
and tables for the research paper.

Author: Research Assistant
Date: October 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json

# Set publication-quality plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

sns.set_palette("colorblind")

class ResultsAnalyzer:
    """Analyze and visualize experimental results"""

    def __init__(self, results_dir, output_dir):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.exp1_data = []
        self.exp2_data = []
        self.exp3_data = []
        self.exp4_data = []

    def load_all_results(self):
        """Load all experiment results"""
        print("Loading experiment results...")

        # Experiment 1: Policy Type Comparison
        print("  Loading Experiment 1 (Policy Type Comparison)...")
        for file in self.results_dir.glob('exp1_*_eval.csv'):
            df = pd.read_csv(file)
            # Extract metadata from filename
            # Filename format: exp1_{policy_type}_{cases}cases_eval.csv
            # where policy_type can be: p1_only, p2_only, both
            filename = file.stem.replace('_eval', '')  # Remove _eval suffix

            # Find the last occurrence of a number followed by 'cases'
            import re
            match = re.search(r'(\d+)cases$', filename)
            if not match:
                continue
            cases = match.group(1)

            # Extract policy type by removing exp1_ prefix and _{cases}cases suffix
            policy_type = filename.replace('exp1_', '').replace(f'_{cases}cases', '')

            for _, row in df.iterrows():
                self.exp1_data.append({
                    'policy_config': policy_type.replace('_', ' ').title(),
                    'cases': int(cases),
                    'approach': row['approach'],
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1_score': row['f1_score'],
                    'true_positives': row['true_positives'],
                    'false_positives': row['false_positives'],
                    'false_negatives': row['false_negatives'],
                    'true_negatives': row['true_negatives']
                })

        # Experiment 2: Violation Rate Sensitivity
        print("  Loading Experiment 2 (Violation Rate Sensitivity)...")
        for file in self.results_dir.glob('exp2_*_eval.csv'):
            df = pd.read_csv(file)
            # Extract violation rate from filename
            parts = file.stem.split('_')
            rate_pct = int(parts[2].replace('pct', ''))
            rate = rate_pct / 100.0

            for _, row in df.iterrows():
                self.exp2_data.append({
                    'violation_rate': rate,
                    'approach': row['approach'],
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1_score': row['f1_score']
                })

        # Experiment 3: Ground Truth
        print("  Loading Experiment 3 (Ground Truth)...")
        file = self.results_dir / 'exp3_ground_truth_eval.csv'
        if file.exists():
            df = pd.read_csv(file)
            self.exp3_data = df.to_dict('records')

        # Experiment 4: Scalability
        print("  Loading Experiment 4 (Scalability)...")
        for file in self.results_dir.glob('exp4_*_eval.csv'):
            df = pd.read_csv(file)
            parts = file.stem.split('_')
            cases = int(parts[2].replace('cases', ''))

            # We'll need to add timing info from logs
            # For now, store the data
            for _, row in df.iterrows():
                self.exp4_data.append({
                    'cases': cases,
                    'approach': row['approach'],
                    'f1_score': row['f1_score']
                })

        print(f"Loaded {len(self.exp1_data)} Exp1 records")
        print(f"Loaded {len(self.exp2_data)} Exp2 records")
        print(f"Loaded {len(self.exp3_data)} Exp3 records")
        print(f"Loaded {len(self.exp4_data)} Exp4 records")

    def create_figure_1_policy_comparison(self):
        """Figure 1: Policy Type Comparison (Exp1)"""
        print("\nGenerating Figure 1: Policy Type Comparison...")

        if not self.exp1_data:
            print("  No data for Figure 1")
            return

        df = pd.DataFrame(self.exp1_data)
        df = df[df['approach'] == 'Policy-Aware']  # Focus on our approach

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        metrics = ['precision', 'recall', 'f1_score']
        titles = ['Precision', 'Recall', 'F1-Score']

        for ax, metric, title in zip(axes, metrics, titles):
            pivot = df.pivot_table(values=metric, index='policy_config', columns='cases', aggfunc='mean')

            pivot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{title} by Policy Type')
            ax.set_xlabel('Policy Configuration')
            ax.set_ylabel(title)
            ax.set_ylim(0, 1.1)
            ax.legend(title='Cases', labels=[f'{int(c)} cases' for c in pivot.columns])
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure1_policy_comparison.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure1_policy_comparison.png', bbox_inches='tight')
        plt.close()

        print(f"  Saved: {self.output_dir / 'figure1_policy_comparison.pdf'}")

    def create_figure_2_violation_sensitivity(self):
        """Figure 2: Violation Rate Sensitivity (Exp2)"""
        print("\nGenerating Figure 2: Violation Rate Sensitivity...")

        if not self.exp2_data:
            print("  No data for Figure 2")
            return

        df = pd.DataFrame(self.exp2_data)

        fig, ax = plt.subplots(figsize=(8, 5))

        for approach in df['approach'].unique():
            data = df[df['approach'] == approach]
            data = data.sort_values('violation_rate')

            ax.plot(data['violation_rate'] * 100, data['precision'],
                   marker='o', label=f'{approach} - Precision', linestyle='-')
            ax.plot(data['violation_rate'] * 100, data['recall'],
                   marker='s', label=f'{approach} - Recall', linestyle='--')
            ax.plot(data['violation_rate'] * 100, data['f1_score'],
                   marker='^', label=f'{approach} - F1-Score', linestyle=':')

        ax.set_xlabel('Violation Rate (%)')
        ax.set_ylabel('Score')
        ax.set_title('Performance vs Violation Injection Rate')
        ax.legend(loc='best', frameon=True)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure2_violation_sensitivity.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure2_violation_sensitivity.png', bbox_inches='tight')
        plt.close()

        print(f"  Saved: {self.output_dir / 'figure2_violation_sensitivity.pdf'}")

    def create_figure_3_confusion_matrices(self):
        """Figure 3: Confusion Matrices Comparison (Exp3)"""
        print("\nGenerating Figure 3: Confusion Matrices...")

        if not self.exp3_data:
            print("  No data for Figure 3")
            return

        df = pd.DataFrame(self.exp3_data)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for i, approach in enumerate(df['approach'].unique()):
            row = df[df['approach'] == approach].iloc[0]

            cm = np.array([
                [row['true_negatives'], row['false_positives']],
                [row['false_negatives'], row['true_positives']]
            ])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Predicted Negative', 'Predicted Positive'],
                       yticklabels=['Actual Negative', 'Actual Positive'],
                       cbar_kws={'label': 'Count'})

            axes[i].set_title(f'{approach}\nF1={row["f1_score"]:.3f}')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure3_confusion_matrices.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure3_confusion_matrices.png', bbox_inches='tight')
        plt.close()

        print(f"  Saved: {self.output_dir / 'figure3_confusion_matrices.pdf'}")

    def create_figure_4_scalability(self):
        """Figure 4: Scalability Analysis (Exp4)"""
        print("\nGenerating Figure 4: Scalability Analysis...")

        if not self.exp4_data:
            print("  No data for Figure 4")
            return

        df = pd.DataFrame(self.exp4_data)
        df_policy_aware = df[df['approach'] == 'Policy-Aware']

        fig, ax = plt.subplots(figsize=(8, 5))

        # Group by cases and plot
        summary = df_policy_aware.groupby('cases')['f1_score'].mean().reset_index()

        ax.plot(summary['cases'], summary['f1_score'], marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Cases')
        ax.set_ylabel('F1-Score')
        ax.set_title('Performance Across Dataset Sizes')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.1)

        # Add linear fit annotation
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(summary['cases'], summary['f1_score'])
        ax.text(0.05, 0.95, f'R² = {r_value**2:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure4_scalability.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure4_scalability.png', bbox_inches='tight')
        plt.close()

        print(f"  Saved: {self.output_dir / 'figure4_scalability.pdf'}")

    def generate_summary_table(self):
        """Generate LaTeX summary table"""
        print("\nGenerating Summary Table...")

        if not self.exp1_data:
            print("  No data for summary table")
            return

        df = pd.DataFrame(self.exp1_data)
        df = df[df['cases'] == 1000]  # Use largest dataset

        # Create summary
        summary = df.groupby(['policy_config', 'approach'])[['precision', 'recall', 'f1_score']].mean().reset_index()

        latex_table = summary.to_latex(index=False, float_format="%.4f")

        table_file = self.output_dir / 'table1_summary.tex'
        with open(table_file, 'w') as f:
            f.write("% Summary of Policy Detection Performance\n")
            f.write(latex_table)

        print(f"  Saved: {table_file}")

    def generate_all_visualizations(self):
        """Generate all figures and tables"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        self.load_all_results()

        self.create_figure_1_policy_comparison()
        self.create_figure_2_violation_sensitivity()
        self.create_figure_3_confusion_matrices()
        self.create_figure_4_scalability()
        self.generate_summary_table()

        print("\n" + "="*80)
        print("VISUALIZATION GENERATION COMPLETE")
        print("="*80)
        print(f"\nAll figures saved to: {self.output_dir}")
        print("\nGenerated files:")
        for f in sorted(self.output_dir.glob('*')):
            print(f"  - {f.name}")

def main():
    """Main entry point"""
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results'
    output_dir = base_dir / 'paper_sections'

    if not results_dir.exists() or not any(results_dir.glob('exp*.csv')):
        print("ERROR: No experiment results found!")
        print(f"Expected results in: {results_dir}")
        print("\nPlease run experiments first:")
        print("  python experiments/run_experiments.py")
        return 1

    analyzer = ResultsAnalyzer(results_dir, output_dir)
    analyzer.generate_all_visualizations()

    print("\nNext steps:")
    print("1. Review figures in paper_sections/")
    print("2. Copy paper_sections/ to your Overleaf project")
    print("3. Include figures in your LaTeX document")

    return 0

if __name__ == '__main__':
    sys.exit(main())
