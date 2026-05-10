"""
End-to-end ISIC 2018 lesion analysis pipeline.

Chains a Mask R-CNN segmentation model, localizing and masking the lesions, with an EfficientNet classification model,
predicting the disease class from the cropped lesion region.

Reference datasets:
  Task 1: Codella et al., arXiv:1902.03368, 2019 (segmentation)
  Task 3: Tschandl et al., Scientific Data, 2018 (classification)
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

__all__ = ["ISICPipeline", "PredicctionResult", "EvaluationResult"]

# -----------------------------------------------------------------------------------------------------------
# Return types
# -----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionResult:
    """
    ISICPipeline.predict() output for a single image.
    Attributes:
        mask: Binary predicted lesion mask, shape (H, W), dtype bool.
              All zero if no detection passed the score threshold.
        class_label: Predicted class index in [0, C-1]. -1 if detection failed.
        class_probabilities: Softmax probabilities, shape (C,), dtype float32. None if detection failed.
        detection_failed: True of Mask R-CNN produced no detection > score_threshold. Jaccard will be 0.0 and
                          classification metrics exclude this image.
        score: Mask R-CNN confidence score of the selected detection. 0.0 if detection failed.
    """
    mask: np.ndarray
    class_label: int
    class_probabilities: np.ndarray | None
    detection_failed: bool
    score: float

@dataclass(frozen=True)
class EvaluationResult:
    """
    ISICPipeline.evaluate() output over a validation set.
    Attributes:
        mean_jaccard: Mean threshold Jaccard (T=jaccard_threshold) over all images. Detection failure contributes 0.0
    """




