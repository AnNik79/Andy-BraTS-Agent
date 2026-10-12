"""Beta 2 observational uncertainty and expert-trust analysis.

These utilities consume frozen Beta 1 outputs. They do not alter expert,
controller, inference, or checkpoint behavior.
"""

from .probability import (
    ExpertProbabilityBatch,
    masked_mean_probability,
    stack_expert_probabilities,
    validate_probability_tensor,
)
from .disagreement import (
    analyze_uncertainty,
    entropy_map,
    summarize_uncertainty_maps,
)
from .expert_trust import analyze_expert_trust

__all__ = [
    "ExpertProbabilityBatch",
    "analyze_expert_trust",
    "analyze_uncertainty",
    "entropy_map",
    "masked_mean_probability",
    "stack_expert_probabilities",
    "summarize_uncertainty_maps",
    "validate_probability_tensor",
]
