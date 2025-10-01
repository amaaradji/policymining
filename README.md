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
│   └── test_data.csv          # Sample test data
├── legacy_research_code/      # Original research implementations (archived)
│   ├── resource_availability_policy.py   # Original paper implementation
│   └── policy_mining.py                  # Early policy mining utilities
├── paper_sections/             # Research paper materials
│   ├── section5_updated.tex   # LaTeX evaluation section
│   └── *.png, *.txt          # Figures and experimental results
├── data/                      # Event log datasets (BPI2017)
├── results/                   # Generated analysis results
└── README.md                  # This file (comprehensive documentation)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/amaaradji/policymining.git
cd policymining

# Install dependencies
pip install pm4py pandas matplotlib numpy seaborn scikit-learn pyyaml
```

### Download BPI Challenge 2017 Dataset

For research experiments, download the BPIC 2017 dataset:

```bash
# Download from 4TU.ResearchData
cd data
wget https://data.4tu.nl/ndownloader/items/34c3f44b-3101-4ea9-8281-e38905c68b8d/versions/1 -O "BPI Challenge 2017_1_all.zip"

# Extract the dataset
unzip "BPI Challenge 2017_1_all.zip"

# Verify the main event log exists
ls -lh "BPI Challenge 2017.xes"
```

**Alternative**: Download manually from https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b and extract to `data/` directory.

**Dataset Info**: 31,509 loan applications, 1.2M+ events, Feb 2016 - Feb 2017

## Usage

### Basic Policy Checking

Run policy checking on an event log:

```bash
cd policy_engine
python policy_engine.py \
  --events test_data.csv \
  --config config/config.yaml \
  --out policy_log.csv \
  --verbose
```

### Evaluation Mode

Test detection performance with synthetic violation injection:

```bash
python policy_engine.py \
  --events test_data.csv \
  --config config/config.yaml \
  --out policy_log.csv \
  --evaluate \
  --violation-rate 0.05 \
  --eval-out evaluation_results.csv
```

**CLI Parameters:**
- `--events`: Path to event log (CSV or XES format)
- `--config`: Path to YAML configuration file
- `--out`: Output path for policy log CSV
- `--roles`: (Optional) Path to resource roles CSV
- `--verbose`: Enable detailed logging
- `--evaluate`: Enable evaluation mode with synthetic violations
- `--violation-rate`: Proportion of events to violate (default: 0.05)
- `--eval-out`: Save evaluation metrics to CSV

## Testing the Code

> **For Research Paper Evaluation**: See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for comprehensive experimental framework with battery of tests, automated figure generation, and Overleaf packaging.

### Quick Test with Sample Data

Test the policy engine with provided sample data:

```bash
cd policy_engine

# Test basic functionality
python policy_engine.py \
  --events test_data.csv \
  --config config/config.yaml \
  --out test_output.csv \
  --verbose

# Verify output
head test_output.csv
```

**Expected output**: Policy log with 10 entries (5 cases × 2 policies)

### Test with BPIC 2017 Dataset

Run policy checking on the full research dataset:

```bash
cd policy_engine

# Run on BPIC 2017 dataset (subset for faster testing)
python policy_engine.py \
  --events ../data/"BPI Challenge 2017.xes" \
  --config config/config.yaml \
  --out ../results/bpic2017_policy_log.csv \
  --verbose

# For full dataset analysis with evaluation
python policy_engine.py \
  --events ../data/"BPI Challenge 2017.xes" \
  --config config/config.yaml \
  --out ../results/bpic2017_policy_log.csv \
  --evaluate \
  --violation-rate 0.05 \
  --eval-out ../results/bpic2017_evaluation.csv \
  --verbose
```

**Note**: Full dataset processing may take several minutes depending on system resources.

### Test with Evaluation Mode

Test synthetic violation injection and detection:

```bash
python policy_engine.py \
  --events test_data.csv \
  --config config/config.yaml \
  --out test_policy_log.csv \
  --evaluate \
  --violation-rate 0.1 \
  --eval-out test_eval_results.csv
```

Check evaluation results:
```bash
cat test_eval_results.csv
```

Expected metrics: precision, recall, F1-score for policy-aware detection

### Test with Custom Data

Prepare your event log CSV with required columns:
- `case_id`: Unique case identifier
- `activity`: Activity name
- `timestamp`: ISO format timestamp (e.g., `2017-03-01T08:00:00Z`)
- `performer`: Resource who performed the activity
- `role`: Resource role (e.g., `SENIOR`, `JUNIOR`, `CLERK`)
- `amount`: Transaction amount (for P1 policy)
- `seq`: Sequence number within case

Example:
```csv
case_id,activity,timestamp,performer,role,amount,seq
APP_001,A_SUBMITTED,2017-03-01T08:00:00Z,usr_1,CLERK,25000,1
APP_001,O_ACCEPTED,2017-03-02T10:34:12Z,usr_45,CLERK,25000,2
```

Run the engine:
```bash
python policy_engine.py \
  --events your_data.csv \
  --config config/config.yaml \
  --out your_policy_log.csv
```

## Configuration

Policies are configured via YAML file (`policy_engine/config/config.yaml`):

```yaml
# Enable/disable policies
enabled_policies: ["P1", "P2"]

# P1: Senior Approval Policy
thresholds:
  P1_T: 20000                 # Amount threshold requiring senior approval
  P1_delegate_wait_h: 24      # Hours before delegation allowed

activities:
  TARGET_ACTS: ["O_ACCEPTED", "A_APPROVED"]
  APPROVAL_ACTS: ["W_Validate application", "A_Accepted"]
  REQUEST_ANCHORS: ["A_SUBMITTED"]

roles:
  senior_regex: "(?i)(SENIOR|MANAGER)"

# P2: Resource Availability Policy
availability:
  default_start_hour: 9       # Work start (9 AM)
  default_end_hour: 17        # Work end (5 PM)
  default_days: [0, 1, 2, 3, 4]  # Mon-Fri (0=Monday, 6=Sunday)

  # Optional: Per-resource custom windows
  resource_windows:
    "User_123":
      start_hour: 14
      end_hour: 22
      allowed_days: [0, 1, 2, 3, 4, 5]

defaults:
  timezone: "UTC"
  unknown_role_is_junior: true
```

## Architecture

The framework follows a modular, extensible architecture:

### Core Components

1. **Policy Interface** (`Policy` abstract class)
   - `prepare_case(case_df, context)`: Precompute case-level data
   - `evaluate_event(event_row, case_state, context)`: Evaluate individual events

2. **Policy Implementations**
   - **P1 - SeniorApprovalPolicy**: Validates senior approval with delegation support
     - Checks if high-value cases (amount ≥ threshold) have senior approval
     - Allows junior approval after configurable wait time (delegation)
     - Outcomes: `duty_met`, `duty_met_via_delegation`, `duty_unmet`, `not_applicable`

   - **P2 - ResourceAvailabilityPolicy**: Validates working hours/days
     - Checks events occur within resource availability windows
     - Supports per-resource custom schedules
     - Outcomes: `duty_met`, `duty_unmet`

3. **Policy Engine** (`PolicyEngine` class)
   - Loads event logs (CSV/XES formats)
   - Applies multiple policies concurrently
   - Generates unified policy log output

4. **Evaluation Framework** (`evaluation.py`)
   - Synthetic violation injection for testing
   - Ground truth comparison
   - Performance metrics (precision, recall, F1)

### Research Methodology

**Policy-Aware vs Event-Log-Only Detection**:

| Approach | Method | Strengths | Limitations |
|----------|--------|-----------|-------------|
| **Event-Log-Only** | Infers "normal" patterns using statistical methods | No external dependencies | Misses ~50% of violations; cannot detect statistically "normal" violations |
| **Policy-Aware** | Uses explicit policy definitions to check compliance | 100% recall; detects all policy violations | Requires explicit policy definitions |

**Key Finding**: Policy-Aware approach achieves perfect recall (100%) while Event-Log-Only misses nearly half of violations.

## Extending the Framework

Add custom policies by implementing the `Policy` interface:

### Step 1: Create Policy Class

```python
# In policy_engine/policy_engine.py

class MyCustomPolicy(Policy):
    """Custom policy implementation"""

    def __init__(self, policy_id: str, config: Dict):
        super().__init__(policy_id, config)
        # Load policy-specific configuration
        self.threshold = config['my_policy'].get('threshold', 100)

    def prepare_case(self, case_df: pd.DataFrame, context: Dict) -> Dict:
        """Precompute case-level information"""
        # Example: Find first event timestamp
        first_event = case_df['timestamp'].min()
        return {'first_event': first_event}

    def evaluate_event(self, event_row: pd.Series, case_state: Dict,
                      context: Dict) -> Optional[Dict]:
        """Evaluate event against policy"""
        # Skip if not target activity
        if event_row['activity'] != 'TARGET_ACTIVITY':
            return None

        # Check policy condition
        is_compliant = self.check_compliance(event_row, case_state)

        # Return policy log entry
        return {
            'case_id': event_row['case_id'],
            'seq': int(event_row['seq']),
            'event_activity': event_row['activity'],
            'event_ts': event_row['timestamp'].isoformat(),
            'performer': event_row['performer'],
            'policy_id': self.policy_id,
            'rule_id': f"{self.policy_id}.my_rule",
            'outcome': 'duty_met' if is_compliant else 'duty_unmet',
            'evidence': f"threshold={self.threshold}"
        }
```

### Step 2: Register Policy

In `_initialize_policies()` method:

```python
if 'P3' in enabled_policies:
    policies.append(MyCustomPolicy("P3", self.config))
```

### Step 3: Add Configuration

In `config/config.yaml`:

```yaml
enabled_policies: ["P1", "P2", "P3"]

my_policy:
  threshold: 100
  # Additional parameters
```

## Repository Contents

### Main Implementation

- **`policy_engine/`** - Unified policy framework (recommended)
  - `policy_engine.py` - Core engine with P1 (Senior Approval) and P2 (Availability) policies
  - `evaluation.py` - Evaluation framework with synthetic violation injection
  - `config/config.yaml` - Policy configuration (thresholds, activities, availability windows)
  - `test_data.csv` - Sample data (5 cases, 30 events) for quick testing

### Research Materials

- **`data/`** - Event log datasets
  - `BPI Challenge 2017.xes` - BPIC 2017 dataset (31K cases, 1.2M events) - **download required**
  - `README.md` - Dataset download instructions and statistics
  - `.gitignore` - Excludes large data files from version control

- **`paper_sections/`** - Research paper LaTeX sections and figures
  - `section5_updated*.tex` - Evaluation sections for paper
  - `*.png` - Violation detection visualizations (precision/recall/F1)
  - `*.txt` - Experimental result summaries

- **`results/`** - Generated analysis outputs
  - Policy logs (CSV format)
  - Evaluation metrics
  - Visualizations (created by running experiments)

### Archived Code

- **`legacy_research_code/`** - Original research implementations
  - Preserved for reproducibility of original paper experiments
  - Contains specialized visualization and event-log-only comparison features
  - Not recommended for new projects - use `policy_engine/` instead

## Research Paper

**"Complementing Event Logs with Policy Logs for Business Process Mining"**

This implementation demonstrates that policy-aware conformance checking significantly outperforms event-log-only methods:
- **Policy-Aware**: 100% recall, perfect violation detection
- **Event-Log-Only**: ~50% recall, misses violations that appear statistically "normal"

Paper sections and experimental results are available in `paper_sections/`.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- BPI Challenge 2017 for the dataset
- PM4Py team for the process mining library

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{policymining2025,
  title={Complementing Event Logs with Policy Logs for Business Process Mining},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```
