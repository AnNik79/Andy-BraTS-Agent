"""Observational expert-analysis instrumentation.

Segmentation error = disagreement with ground truth.
Expert disagreement = disagreement among experts.
These are not the same quantity.

Nothing in this package trains models, edits voxels, or changes production
forward / debate / controller / LLM behavior. Fitting representation statistics
is allowed only on expert_train.
"""

from .extract import analyze_segmentation_expert, load_frozen_cnn, representation_spec
from .pooling import pool_patch, pool_patient, pool_region
from .error_maps import segmentation_error_maps
from .pairwise import pairwise_expert_analysis, SEGMENTATION_PAIRS
from .correlation import error_correlation_matrix, prediction_similarity_matrix, export_matrices
from .svd import latent_svd
from .mahalanobis import ExpertMahalanobis
from .schema import empty_region_record, REGION_RECORD_FIELDS
from .difficulty import novelty_disagreement_quadrant
from .active_learning import unlabeled_region_priority
from .tensor_layout import ANALYSIS_AXES
