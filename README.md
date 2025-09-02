# Policy Mining Framework with Resource Availability Constraints

This repository contains an implementation of a policy-aware conformance checking framework for business process mining, with a focus on resource availability constraints. The framework demonstrates how policy logs can complement event logs to detect violations that cannot be reliably identified using event logs alone.

## Overview

The framework augments traditional event logs with policy logs to detect policy violations such as:
- Resource availability violations (working outside permitted hours/days)
- Resource shareability violations
- Action retriability violations

This implementation focuses specifically on resource availability constraints to clearly demonstrate the contrast between event-log-only and policy-aware approaches.

## Features

- Load and preprocess event logs (supports BPI2017 dataset)
- Define and extract resource availability policies
- Augment event logs with synthetic policy violations
- Generate explicit policy logs with availability constraints
- Detect violations using both event-log-only and policy-aware approaches
- Evaluate detection performance with precision, recall, and F1-score
- Visualize results with comprehensive plots and metrics

## Repository Structure

```
.
├── code/
│   ├── resource_availability_policy.py   # Main implementation
│   └── utils/                           # Utility modules
├── data/
│   └── BPI Challenge 2017.xes           # BPI2017 dataset
├── results/
│   ├── augmented_event_log.csv          # Event log with injected violations
│   ├── availability_policy_log.xes      # Generated policy log
│   ├── availability_analysis_results.csv # Complete analysis results
│   ├── availability_summary_results.txt # Summary of key findings
│   ├── availability_violation_detection_results.png # Performance visualization
│   └── violations_by_resource.png       # Resource violation rates
├── section5_updated.tex                 # Updated evaluation section for paper
└── README.md                            # This file
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

3. Download the BPI2017 dataset from the 4TU.ResearchData repository:
   https://data.4tu.nl/ndownloader/items/34c3f44b-3101-4ea9-8281-e38905c68b8d/versions/1

## Usage

To run the resource availability policy mining framework:

```bash
python code/resource_availability_policy.py
```

By default, the script analyzes a subset of the BPI2017 dataset (200 traces). You can modify the `max_traces` parameter in the main function to control the number of traces to analyze.

## Methodology

### Data Augmentation

The framework augments the event log with synthetic policy violations by:
1. Defining resource availability windows (e.g., 9AM-5PM, Monday-Friday)
2. Randomly selecting events (5% by default)
3. Modifying timestamps to fall outside availability windows
4. Preserving original timestamps for evaluation

### Detection Methods

Two approaches are implemented and compared:

1. **Event Log Only (Baseline)**:
   - Infers "normal" working hours using statistical methods
   - Identifies events outside these inferred hours as violations

2. **Policy-Aware (Proposed)**:
   - Uses explicit policy definitions from policy logs
   - Checks each event against defined availability windows
   - Flags events outside windows as violations

## Results

Key findings from our analysis:

- Event Log Only method: Precision=0.1908, Recall=0.5214, F1=0.2794
- Policy-Aware method: Precision=0.1207, Recall=1.0000, F1=0.2154

The Policy-Aware approach achieves perfect recall, detecting all violations, while the Event Log Only approach misses nearly half of the violations.

## Paper

This implementation supports the research paper "Complementing Event Logs with Policy Logs for Business Process Mining," specifically the evaluation section that demonstrates the value of policy logs for conformance checking.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BPI Challenge 2017 for providing the dataset
- PM4Py team for the process mining library
