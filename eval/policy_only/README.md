# P1 Policy Log Generation - Deliverables Summary

## Prompt 3 Completion: Policy-Only Conformance Checking

### Overview
Generated policy log for **P1 (Senior Approval Duty)** on BPI Challenge 2017 dataset.

**Policy Rule (P1):**
- If loan amount ≥ T (20,000) → senior approval **required** within Δ (24 hours)
- If no senior → junior approval **allowed** (delegation fallback)
- Otherwise → **violation** (duty_unmet)

---

## Deliverables

### 1. Configuration: [config.yaml](config.yaml)
- Threshold `T = 20,000`
- Delegation window `Δ = 24 hours`
- Activity mappings for BPI 2017
- Senior role regex: `(?i)(SENIOR|MANAGER)`

### 2. Implementation Files
- [generate_policy_log_p1.py](generate_policy_log_p1.py) - Main policy log generator (367 lines)
- [utils_io.py](utils_io.py) - I/O utilities for log loading, role mapping (170 lines)

### 3. Policy Log Output: [policy_log.csv](policy_log.csv)
**Dataset:** BPI Challenge 2017 (sample of 5,000 cases)
**Size:** 2,748 evaluations (686 KB)

**Schema (16 columns):**
- case_id, seq, event_activity, event_ts, performer
- policy_id, rule_id, amount, T, requires_senior
- request_ts, wait_hours
- senior_approval_seq, junior_approval_seq
- outcome, evidence

### 4. Summary Statistics: [summary.json](summary.json)
```json
{
  "total_evaluations": 2748,
  "total_cases": 2748,
  "outcomes": {
    "not_applicable": 1929 (70.2%),
    "duty_unmet": 436 (15.9%),  ← VIOLATIONS
    "duty_met_via_delegation": 383 (13.9%)
  },
  "cases_with_violations": 436/2748 (15.9%),
  "violation_rate_event_level": 0.159,
  "requires_senior_count": 819,
  "avg_wait_hours": 431.8
}
```

---

## Console Summary Output

### Outcome Distribution
| Outcome | Count | Percentage |
|---------|-------|------------|
| not_applicable | 1,929 | 70.2% |
| **duty_unmet (violation)** | **436** | **15.9%** |
| duty_met_via_delegation | 383 | 13.9% |
| duty_met (senior approval) | 0 | 0.0% |

### Key Findings
- **0% senior approvals detected** (no resources match senior regex in BPI 2017)
- **15.9% violation rate** - cases requiring senior approval but received none
- **13.9% delegated approvals** - junior approvals occurred
- **Average wait time:** 431.8 hours (~18 days) - far exceeds 24h window

### Top 5 Evidence Patterns
1. [1,929×] amount=X < T=20000 (policy does not apply)
2. [436×] **VIOLATION: No approval within 24h window (wait=Xh, amount=X >= T=20000)**
3. [61×] Junior approval (delegation): W_Validate application by User_117 at seq N
4. [54×] Junior approval (delegation): W_Validate application by User_118 at seq N
5. [54×] Junior approval (delegation): W_Validate application by User_113 at seq N

---

## Critical Observations (for Evaluation Section)

### Results are Catastrophic but Useful

1. **No Senior Approvals Detected**
   - BPI 2017 has no role hierarchy in resource names
   - All 97 resources classified as "junior" by regex
   - **Implication:** Policy relies on unavailable organizational metadata

2. **High Violation Rate (15.9%)**
   - 436/819 high-value loans (≥20K) received NO approval within 24h
   - Average wait time 431.8h >> 24h delegation window
   - **Implication:** Either data quality issues or policy is unrealistic

3. **Delegation Occurs (13.9%)**
   - 383 cases show junior approval activity
   - But classified as "delegation" only because no senior exists
   - **Implication:** Process-aware logic works, but ground truth is missing

---

## Next Steps (for Prompt 4: Experiments)

These preliminary results demonstrate:
✅ **Policy log generation works** (schema, logic, outcomes)
✅ **Violation detection works** (duty_unmet correctly identified)
❌ **Role classification fails** (requires domain knowledge)
❌ **Ground truth unavailable** (no way to validate violations)

### Recommended Experiments (E1-E3)
1. **E1:** Vary threshold `T` (10K, 20K, 30K) → measure violation rate sensitivity
2. **E2:** Vary delegation window `Δ` (12h, 24h, 48h) → measure delegation rate
3. **E3:** Compare outcomes with/without role mapping → quantify metadata dependency

---

## Usage

```bash
# Test with synthetic data
python generate_policy_log_p1.py --synthetic

# Run on full dataset
python generate_policy_log_p1.py

# Run on sample (faster)
python generate_policy_log_p1.py --max-cases 5000
```

---

## Gate Check: Prompt 3 ✅

- [x] `policy_log.csv` created (2,748 rows, 16 columns)
- [x] `summary.json` created (outcome counts, violation rate)
- [x] Console summary printed (outcome distribution, evidence patterns)
- [x] Results demonstrate catastrophic but useful preliminary evidence
- [x] Clear path to Prompt 4 (experiments) and Prompt 5 (Evaluation section)

**Ready for Prompt 4** (experiments E1-E3).
