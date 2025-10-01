# Policy Engine Documentation

## Overview

This policy engine implements a general, policy-only conformance checking tool that scans event logs and emits synchronized policy logs. The framework supports multiple policy types and provides an extensible architecture for adding new policies.

## Supported Policies

### Policy P1: Senior Approval with Delegation
"Senior approval if amount ≥ T; delegation allowed after 24h."

Checks whether high-value cases (amount ≥ threshold) receive appropriate approval from senior resources, with support for delegation after a waiting period.

### Policy P2: Resource Availability Constraints
"Resources must work within their defined availability windows."

Validates that events occur within permitted working hours and days for each resource. Supports custom availability windows per resource.

## Architecture

The policy engine follows a modular architecture:

1. **PolicyEngine**: Core engine that processes event logs and applies policies
2. **Policy Interface**: Abstract base class for all policy implementations
3. **Policy Implementations**: P1 (Senior Approval), P2 (Resource Availability)
4. **Evaluation Framework**: Tools for synthetic violation injection and performance measurement

## Configuration

The engine is configured via a YAML file with the following parameters:

```yaml
# Enabled policies
enabled_policies: ["P1", "P2"]

# P1: Senior Approval Policy
thresholds:
  P1_T: 20000               # Amount threshold for senior approval
  P1_delegate_wait_h: 24    # Hours to wait before delegation is allowed

activities:
  TARGET_ACTS: ["O_ACCEPTED", "A_APPROVED"]
  APPROVAL_ACTS: ["W_Validate application", "A_Accepted"]
  REQUEST_ANCHORS: ["A_SUBMITTED"]

roles:
  senior_regex: "(?i)(SENIOR|MANAGER)"

# P2: Resource Availability Policy
availability:
  default_start_hour: 9     # Default work start (9 AM)
  default_end_hour: 17      # Default work end (5 PM)
  default_days: [0, 1, 2, 3, 4]  # Monday-Friday
  resource_windows: {}      # Per-resource custom windows

defaults:
  timezone: "UTC"
  unknown_role_is_junior: true
```

## Policy P1 Implementation

Policy P1 implements the rule: "Senior approval if amount ≥ T; delegation allowed after 24h."

The policy checks:
1. If the loan amount is ≥ T (configurable threshold)
2. If a senior resource has approved the application
3. If 24 hours have passed since submission, allowing junior approval (delegation)

## Usage

### Basic Usage

Run policy checking on an event log:

```bash
python policy_engine.py --events <event_log.csv> --config config.yaml --out policy_log.csv
```

With optional resource roles:

```bash
python policy_engine.py --events events.csv --roles roles.csv --config config.yaml --out policy_log.csv --verbose
```

### Evaluation Mode

Run with synthetic violation injection to evaluate detection performance:

```bash
python policy_engine.py --events events.csv --config config.yaml --out policy_log.csv --evaluate --violation-rate 0.05 --eval-out evaluation_results.csv
```

Parameters:
- `--evaluate`: Enable evaluation mode with synthetic violations
- `--violation-rate`: Proportion of events to violate (default: 0.05)
- `--eval-out`: Path to save evaluation metrics

## Input Format

The event log CSV should have the following columns:
- case_id: Unique identifier for each case
- activity: Name of the activity
- timestamp: ISO format timestamp
- performer: Resource who performed the activity
- role: Role of the performer (optional)
- amount: Loan amount
- seq: Sequence number within the case

## Output Format

The policy log CSV contains:
- case_id: Case identifier
- seq: Sequence number
- event_activity: Activity name
- event_ts: Event timestamp
- performer: Resource who performed the activity
- policy_id: Identifier of the policy (e.g., P1)
- rule_id: Specific rule within the policy
- amount: Loan amount
- T: Threshold amount
- requires_senior: Whether senior approval is required
- request_ts: Timestamp of the request
- wait_hours: Hours waited since request
- senior_approval_seq: Sequence number of senior approval (if any)
- junior_approval_seq: Sequence number of junior approval (if any)
- outcome: Policy outcome (duty_met, duty_met_via_delegation, not_applicable, duty_unmet)
- evidence: Additional evidence for the outcome

## Example Results

The policy engine correctly identifies:
- Cases where senior approval was required and provided
- Cases where delegation was allowed after 24 hours
- Cases where the policy was not applicable (amount < T)

## Extensibility

The framework is designed to be extensible:

### Adding New Policies

1. Create a new class inheriting from `Policy`:

```python
class MyCustomPolicy(Policy):
    def __init__(self, policy_id: str, config: Dict):
        super().__init__(policy_id, config)
        # Initialize policy-specific parameters

    def prepare_case(self, case_df: pd.DataFrame, context: Dict) -> Dict:
        # Precompute case-level information
        return {}

    def evaluate_event(self, event_row: pd.Series, case_state: Dict, context: Dict) -> Optional[Dict]:
        # Evaluate event and return policy log row or None
        return {...}
```

2. Register the policy in `_initialize_policies()` method:

```python
if 'P3' in enabled_policies:
    policies.append(MyCustomPolicy("P3", self.config))
```

3. Add configuration parameters to `config.yaml`

4. Update `enabled_policies` list to include your new policy

## Integration with Existing Code

This unified policy engine reconciles the original resource availability implementation ([resource_availability_policy.py](../code/resource_availability_policy.py)) with a general policy framework:

- **Resource availability checking**: Now available as Policy P2
- **Senior approval checking**: Implemented as Policy P1
- **Evaluation framework**: Supports both policies with synthetic violation injection
- **Unified interface**: All policies use the same abstract base class and engine
