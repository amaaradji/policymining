#!/usr/bin/env python3
"""
Configuration for synthetic evaluation.
Provides reproducibility via run_id and configurable parameters.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SyntheticConfig:
    """Configuration for synthetic log generation and evaluation."""

    # Reproducibility
    seed: int = 42
    run_id: Optional[str] = None

    # Output directories
    base_output_dir: Path = Path('eval/synthetic/outputs')

    # Synthetic log generation parameters (CP1)
    num_cases: int = 1000
    num_activities: int = 8
    model_type: str = 'simple_process'  # simple_process, purchase_order, loan_application

    # Policy parameters (for CP2 injection)
    senior_approval_threshold: float = 20000.0
    delegation_window_hours: float = 24.0

    # Violation injection rates (CP2)
    violation_rate: float = 0.2  # 20% of applicable cases will have violations

    # Severity scoring weights (Checkpoint 1)
    severity_weight_lateness: float = 0.6  # Weight for lateness component
    severity_weight_role: float = 0.4  # Weight for role penalty component

    # Role penalties for severity calculation
    role_penalty_missing: float = 1.0  # No approval at all
    role_penalty_wrong_role: float = 0.5  # Wrong role (e.g., junior when senior required)
    role_penalty_correct: float = 0.0  # Correct role

    # Noise injection parameters (Checkpoint 2)
    enable_noise: bool = True  # Enable noise scenarios
    near_miss_rate: float = 0.15  # Rate of near-miss cases (approvals at boundary)
    boundary_epsilon_minutes: int = 5  # Minutes near delta boundary for near-misses
    multiple_approval_rate: float = 0.10  # Rate of cases with multiple approvals
    missing_role_rate: float = 0.05  # Rate of events with missing role attribute
    timestamp_jitter_minutes: int = 2  # Max jitter for timestamp noise
    out_of_order_rate: float = 0.02  # Rate of out-of-order event pairs

    def __post_init__(self):
        """Initialize run_id and create output directory."""
        if self.run_id is None:
            self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.output_dir = self.base_output_dir / self.run_id

    def create_output_dir(self) -> Path:
        """Create timestamped output directory for this run."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def get_output_path(self, filename: str) -> Path:
        """Get full path for an output file."""
        return self.output_dir / filename
