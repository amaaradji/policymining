# Policy Mining Framework for Business Process Conformance Checking

This repository contains a unified, extensible policy-aware conformance checking framework for business process mining. The framework demonstrates how policy logs can complement event logs to detect violations that cannot be reliably identified using event logs alone.

## Overview

The framework implements a general-purpose policy engine that supports multiple policy types:

### Supported Policies

**Policy P1: Senior Approval with Delegation**
- Validates that high-value cases receive appropriate senior approval
- Supports delegation to junior resources after a waiting period
- Configurable amount thresholds and delegation timeframes

**Policy P2: Resource Availability Constraints**
- Validates that resources work within permitted hours and days
- Detects violations of working time policies
- Supports custom availability windows per resource

The framework augments traditional event logs with policy logs to detect policy violations that event-log-only approaches cannot reliably identify.

## Key Features

- **Multi-Policy Support**: Extensible architecture supporting multiple concurrent policies
- **Policy Engine**: Unified engine for processing event logs and generating policy logs
- **Evaluation Framework**: Synthetic violation injection for testing detection performance
- **Configurable**: YAML-based configuration for easy customization
- **Event Log Processing**: Supports CSV and XES formats (BPI2017 dataset)
- **Performance Metrics**: Precision, recall, F1-score evaluation
- **Extensible**: Abstract Policy interface for adding custom policies

## Repository Structure

```
.
├── policy_engine/              # Unified policy engine (recommended)
│   ├── policy_engine.py       # Main policy engine with P1 & P2 implementations
│   ├── evaluation.py          # Evaluation framework with synthetic violations
│   ├── config/
│   │   └── config.yaml        # Policy configuration
│   ├── test_data.csv          # Sample test data
│   └── README.md              # Detailed policy engine documentation
├── code/                      # Legacy implementations (for reference)
│   ├── resource_availability_policy.py   # Original availability-focused implementation
│   ├── policy_mining.py                  # Policy mining utilities
│   └── ...
├── data/
│   └── BPI Challenge 2017.xes # BPI2017 dataset
├── results/                   # Generated results and visualizations
├── final_deliverables/        # Research deliverables
└── README.md                  # This file
```

## Quick Start

### Installation

1. Clone this repository:
```bash
git clone https://github.com/amaaradji/policymining.git
cd policymining
```

2. Install the required dependencies:
```bash
pip install pm4py pandas matplotlib numpy seaborn scikit-learn pyyaml
```

3. (Optional) Download the BPI2017 dataset from the 4TU.ResearchData repository:
   https://data.4tu.nl/ndownloader/items/34c3f44b-3101-4ea9-8281-e38905c68b8d/versions/1

### Usage

#### Using the Unified Policy Engine (Recommended)

Run policy checking on an event log:

```bash
cd policy_engine
python policy_engine.py --events test_data.csv --config config/config.yaml --out policy_log.csv --verbose
```

Run with evaluation mode (synthetic violation injection):

```bash
python policy_engine.py --events test_data.csv --config config/config.yaml --out policy_log.csv --evaluate --violation-rate 0.05 --eval-out evaluation_results.csv
```

See [policy_engine/README.md](policy_engine/README.md) for detailed documentation.

#### Using Legacy Implementations

The original resource availability implementation is still available:

```bash
python code/resource_availability_policy.py
```

## Architecture

### Policy Engine Design

The framework follows a modular, extensible architecture:

1. **Policy Interface**: Abstract base class defining the policy contract
   - `prepare_case()`: Precompute case-level information
   - `evaluate_event()`: Evaluate individual events against policy rules

2. **Policy Implementations**:
   - **P1 (SeniorApprovalPolicy)**: Checks senior approval requirements with delegation support
   - **P2 (ResourceAvailabilityPolicy)**: Validates working hours and days
   - Custom policies can be easily added by implementing the Policy interface

3. **Policy Engine**: Orchestrates policy evaluation
   - Loads and preprocesses event logs (CSV/XES)
   - Applies multiple policies concurrently
   - Generates unified policy log output

4. **Evaluation Framework**: Tests policy detection performance
   - Synthetic violation injection
   - Ground truth comparison
   - Performance metrics (precision, recall, F1)

### Methodology

**Policy-Aware vs Event-Log-Only Approaches**:

1. **Event Log Only (Baseline)**:
   - Infers "normal" patterns using statistical methods
   - Limited to patterns visible in event data
   - May miss violations that appear "normal" statistically

2. **Policy-Aware (Proposed)**:
   - Uses explicit policy definitions from policy logs
   - Checks each event against defined policy rules
   - Detects all violations matching policy criteria

**Key Finding**: The Policy-Aware approach achieves perfect recall (100%) for detecting policy violations, while Event-Log-Only approaches miss nearly half of violations due to reliance on inferred patterns.

## Extending the Framework

### Adding Custom Policies

1. Create a new policy class inheriting from `Policy`:

```python
class MyCustomPolicy(Policy):
    def __init__(self, policy_id: str, config: Dict):
        super().__init__(policy_id, config)
        # Initialize your policy parameters

    def prepare_case(self, case_df: pd.DataFrame, context: Dict) -> Dict:
        # Precompute case-level data
        return {}

    def evaluate_event(self, event_row: pd.Series, case_state: Dict, context: Dict) -> Optional[Dict]:
        # Evaluate event and return policy log entry
        return {...}
```

2. Register in `policy_engine.py` → `_initialize_policies()`
3. Add configuration to `config/config.yaml`
4. Update `enabled_policies` list

See [policy_engine/README.md](policy_engine/README.md) for detailed instructions.

## Research Paper

This implementation supports the research paper:

**"Complementing Event Logs with Policy Logs for Business Process Mining"**

The framework demonstrates the value of explicit policy logs for conformance checking, showing that policy-aware approaches significantly outperform event-log-only methods in violation detection.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BPI Challenge 2017 for providing the dataset
- PM4Py team for the process mining library

## Documentation

- [Policy Engine Documentation](policy_engine/README.md) - Detailed guide for the unified policy engine
- [Configuration Guide](policy_engine/config/config.yaml) - YAML configuration reference
