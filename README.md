# Policy Mining and Conformance Checking Framework

This repository contains an implementation of a policy-aware conformance checking framework for business process mining. The framework uses the BPI2017 dataset to detect policy violations based on resource usage and duration constraints.

## Overview

The framework augments traditional event logs with policy logs to detect policy violations such as overuse, unauthorized access, or uncoordinated sharing of resources. It implements a replay-based conformance checking approach that compares the performance of policy-aware methods against traditional event log only methods.

## Features

- Load and preprocess event logs (supports BPI2017 dataset)
- Define and extract policy rules from event logs
- Generate synthetic policy logs with resource-specific thresholds
- Detect policy violations using replay-based conformance checking
- Evaluate detection performance with precision, recall, and F1-score
- Calculate edit distances for trace alignment
- Analyze violations by resource
- Visualize results with comprehensive plots and metrics

## Repository Structure

```
.
├── code/
│   ├── policy_mining.py             # Original implementation
│   └── policy_mining_optimized.py   # Memory-optimized implementation
├── data/
│   └── policy_log.xes               # Generated policy log
├── results/
│   ├── analysis_results_sample.csv  # Sample of analysis results
│   ├── edit_distance_results.csv    # Trace alignment metrics
│   ├── policy_violation_detection_results.png  # Performance visualization
│   ├── summary_results.txt          # Summary of key findings
│   ├── violation_rate_by_resource.png  # Resource violation rates
│   └── violation_types.csv          # Detailed violation statistics
└── README.md                        # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/amaaradji/policymining.git
cd policymining
```

2. Install the required dependencies:
```bash
pip install pm4py pandas matplotlib numpy seaborn scikit-learn
```

## Usage

To run the policy mining framework:

```bash
python code/policy_mining_optimized.py
```

By default, the script analyzes a subset of the BPI2017 dataset. You can modify the `max_traces` parameter in the main function to control the number of traces to analyze.

## Results

The framework generates comprehensive results including:

- Performance metrics (precision, recall, F1-score)
- Trace alignment metrics (edit distance)
- Visualizations of policy violations
- Resource-specific violation statistics

### Key Findings

From our analysis of the BPI2017 dataset:

- Total events analyzed: 7,959
- Total policy violations detected: 431 (5.42%)
- The policy-aware method achieved perfect precision (1.0), recall (1.0), and F1-score (1.0)
- Resources with highest violation rates: User_42, User_35, User_18 (50% each)

## Dataset

This implementation uses the BPI Challenge 2017 dataset, which contains event logs from a loan application process. The dataset is available from the 4TU.ResearchData repository: https://data.4tu.nl/ndownloader/items/34c3f44b-3101-4ea9-8281-e38905c68b8d/versions/1

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BPI Challenge 2017 for providing the dataset
- PM4Py team for the process mining library
