# Policy Engine Documentation

## Overview

This policy engine implements a general, policy-only conformance checking tool that scans event logs and emits synchronized policy logs. The implementation focuses on Policy P1: "Senior approval if amount ≥ T; delegation allowed after 24h."

## Architecture

The policy engine follows a modular architecture:

1. **PolicyEngine**: Core engine that processes event logs and applies policies
2. **Policy Interface**: Abstract base class for all policy implementations
3. **Policy P1**: Implementation of the senior approval and delegation policy

## Configuration

The engine is configured via a YAML file with the following parameters:

```yaml
thresholds:
  P1_T: 20000               # Amount threshold for senior approval
  P1_delegate_wait_h: 24    # Hours to wait before delegation is allowed

activities:
  TARGET_ACTS: ["O_ACCEPTED", "A_APPROVED"]  # Activities that require policy checking
  APPROVAL_ACTS: ["W_Validate application", "W_Complete application", "A_Accepted", "O_Create Offer"]  # Activities that can fulfill approval
  REQUEST_ANCHORS: ["A_SUBMITTED"]  # Activities that start the approval process

roles:
  senior_regex: "(?i)(SENIOR|MANAGER)"  # Regex to identify senior roles

defaults:
  timezone: "UTC"
  unknown_role_is_junior: true  # Default behavior for unknown roles
```

## Policy P1 Implementation

Policy P1 implements the rule: "Senior approval if amount ≥ T; delegation allowed after 24h."

The policy checks:
1. If the loan amount is ≥ T (configurable threshold)
2. If a senior resource has approved the application
3. If 24 hours have passed since submission, allowing junior approval (delegation)

## Usage

```bash
python policy_engine.py --events <event_log.csv> --config <config.yaml> --out <policy_log.csv> --verbose
```

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
1. New policies can be added by implementing the Policy interface
2. Additional rules can be added to existing policies
3. Configuration parameters can be adjusted without code changes
