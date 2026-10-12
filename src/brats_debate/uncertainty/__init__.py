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
from .error_prediction import (
    LogisticErrorModel,
    PlattCalibrator,
    construct_error_target,
    error_prediction_metrics,
    predict_risk_map,
)
from .extraction import (
    VoxelSamplingPolicy,
    compact_expert_outputs,
    extract_sampled_case,
    predict_experts_compact,
    stream_controller_and_uncertainty,
)
from .partitions import make_subject_partitions, validate_subject_partitions

__all__ = [
    "ExpertProbabilityBatch",
    "LogisticErrorModel",
    "PlattCalibrator",
    "VoxelSamplingPolicy",
    "analyze_expert_trust",
    "analyze_uncertainty",
    "compact_expert_outputs",
    "construct_error_target",
    "entropy_map",
    "error_prediction_metrics",
    "extract_sampled_case",
    "make_subject_partitions",
    "masked_mean_probability",
    "predict_risk_map",
    "predict_experts_compact",
    "stack_expert_probabilities",
    "summarize_uncertainty_maps",
    "stream_controller_and_uncertainty",
    "validate_probability_tensor",
    "validate_subject_partitions",
]
