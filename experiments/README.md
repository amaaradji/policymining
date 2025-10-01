# Experimental Evaluation Framework

This directory contains scripts to run comprehensive experiments for the policy mining research paper.

## Overview

The evaluation framework executes a battery of experiments to answer key research questions about policy-aware conformance checking effectiveness, robustness, and scalability.

## Prerequisites

1. **BPIC 2017 Dataset**: Download and extract to `data/` directory
   ```bash
   cd ../data
   wget https://data.4tu.nl/ndownloader/items/34c3f44b-3101-4ea9-8281-e38905c68b8d/versions/1 -O "BPI Challenge 2017_1_all.zip"
   unzip "BPI Challenge 2017_1_all.zip"
   ```

2. **Python Dependencies**:
   ```bash
   pip install pm4py pandas matplotlib seaborn numpy scipy scikit-learn pyyaml
   ```

## Running Experiments

### Step 1: Run All Experiments

Execute the complete experimental battery (this will take time):

```bash
python experiments/run_experiments.py
```

This runs:
- **Experiment 1**: Policy Type Comparison (P1, P2, Both) × (200, 500, 1000 cases)
- **Experiment 2**: Violation Rate Sensitivity (1%, 3%, 5%, 10%, 15%, 20%)
- **Experiment 3**: Ground Truth Comparison (GT1 vs GT2)
- **Experiment 4**: Scalability Analysis (100, 500, 1000, 5000 cases)

**Expected duration**: 30-60 minutes depending on system performance

Results are saved to `results/` directory.

### Step 2: Analyze Results and Generate Figures

Process all experiment results and generate publication-ready figures:

```bash
python experiments/analyze_results.py
```

This generates:
- `paper_sections/figure1_policy_comparison.pdf` - Policy type performance comparison
- `paper_sections/figure2_violation_sensitivity.pdf` - Robustness across violation rates
- `paper_sections/figure3_confusion_matrices.pdf` - Detection effectiveness comparison
- `paper_sections/figure4_scalability.pdf` - Performance across dataset sizes
- `paper_sections/table1_summary.tex` - LaTeX summary table

### Step 3: Package for Overleaf

```bash
python experiments/package_for_overleaf.py
```

This creates `overleaf_package/` with:
- `evaluation.tex` - Complete evaluation section
- All figure files (PDF format for LaTeX)
- Summary table (TEX format)

Simply copy the `overleaf_package/` contents to your Overleaf project.

## Experiment Configurations

### Configuration Files

Located in `experiments/configs/`:

- `config_p1_only.yaml` - Senior Approval policy only
- `config_p2_only.yaml` - Resource Availability policy only
- `config_both.yaml` - Both policies combined

### Customizing Experiments

Edit `run_experiments.py` to:
- Add/remove case size configurations
- Modify violation injection rates
- Change policy configurations
- Add new experiment types

## Results Structure

```
results/
├── exp1_p1_only_200cases_policy_log.csv
├── exp1_p1_only_200cases_eval.csv
├── exp2_rate_01pct_policy_log.csv
├── exp2_rate_01pct_eval.csv
├── exp3_ground_truth_policy_log.csv
├── exp3_ground_truth_eval.csv
├── exp4_scalability_100cases_policy_log.csv
└── exp4_scalability_100cases_eval.csv
```

Each `*_eval.csv` contains:
- `approach`: Detection method (Policy-Aware or Event-Log-Only)
- `precision`, `recall`, `f1_score`: Performance metrics
- `true_positives`, `false_positives`, etc.: Confusion matrix values

## Troubleshooting

**Dataset not found**:
```
ERROR: Dataset not found at data/BPI Challenge 2017.xes
```
→ Download and extract BPIC 2017 dataset to `data/` directory

**Memory errors**:
→ Reduce case limits in `run_experiments.py`
→ Run experiments sequentially instead of all at once

**Missing dependencies**:
```
pip install pm4py pandas matplotlib seaborn numpy scipy scikit-learn pyyaml
```

## Citation

If you use this evaluation framework, please cite:

```bibtex
@article{policymining2025,
  title={Complementing Event Logs with Policy Logs for Business Process Mining},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```
