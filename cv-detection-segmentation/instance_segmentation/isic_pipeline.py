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

__all__ = ["ISICPipeline", "PredictionResult", "EvaluationResult"]

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
        mean_jaccard: Mean threshold Jaccard (T=jaccard_threshold) over all images. Detection failure contributes 0.0.
        balanced_accuracy: Balanced accuracy over images where detection succeeded. None if no image produced a
                           valid detection.
        combined_score: Arithmetic mean of mean_jaccard and balanced_accuracy. None if balanced_accuracy is None.
        n_total: Total number of images evaluated.
        n_detection_failed: Number of images where Mask R-CNN produced no detection.
    """

    mean_jaccard: float
    balanced_accuracy: float | None
    combined_score: float | None
    n_total: int
    n_detection_failed: int


# -----------------------------------------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------------------------------------


class ISICPipeline:
    """
    End-to-end ISIC lesion analysis pipeline.

    Chains Mask R-CNN (segmentation) with EfficientNet (classification):
        1. Mask R-CNN localizes the lesion and produces a binary mask.
        2. The lesion region is cropped using the predicted bounding box.
        3. EfficientNet classifies the crop into a disease class.

    Both models must be preloaded and moved to the target device by the caller. The pipeline device is inferred
    from the first parameter of segment_model.
    All segment_model parameters are assumed to be on the same device.

    Args:
        segment_model: Mask R-CNN. Expects a list containing one FloatTensor of shape (3, H, W) with values in [0,1],
                     and no ImageNet normalization, matching COCO pretraining.
        class_model: EfficientNet. Expects a FloatTensor of shape (1, 3, img_size, img_size), ImageNet-normalised.
        img_size: Spatial size used during classifier training. Default: 224.
        score_threshold: Minimum Mask R-CNN detection confidence to accept. Default: 0.5.
        mask_threshold: Sigmoind threshold for binarising the soft mask output. Default: 0.5.
        jaccard_threshold: Jaccard threshold T below which per-image IoU is set to zero, per the ISIC 2018
                           evaluation protocol (Codella et al., 2019). Default: 0.65.
        class_mean: ImageNet normalization mean applied to classifier crops. Default: (0.485, 0.456, 0.406).
        class_std: ImageNet normalization std applied to classifier crops. Default: (0.229, 0.224, 0.225).
    """

    def __init__(
        self,
        segment_model: nn.Module,
        class_model: nn.Module,
        img_size: int = 224,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        jaccard_threshold: float = 0.65,
        class_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        class_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.segment_model = segment_model.eval()
        self.class_model = class_model.eval()
        self.img_size = img_size
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.jaccard_threshold = jaccard_threshold
        self.class_mean = class_mean
        self.class_std = class_std
        self.device = next(segment_model.parameters()).device

    # --------------------------------------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------------------------------------

    def _run_segmentation(self, image: torch.Tensor) -> tuple[np.ndarray, np.ndarray | None, float]:
        """
        Run Mask R-CNN on a single image.
        Args:
            image: FloatTensor (3, H, W) in [0,1], on self.device

        Returns:
            (binary_mask, box, score) tuple.
            binary_mask: Bool ndarray (W, W). All false if detection failed.
            box: Float bdarrat [x1, y1, x2, y2]. None if failed.
            score: Confidence of selected detection. 0.0 if failed.
        """

        H, W = image.shape[1], image.shape[2]
        output = self.segment_model([image])[0]

        keep = output["scores"] >= self.score_threshold
        if not keep.any():
            return np.zeros((H, W), dtype=bool), None, 0.0

        best_idx = output["scores"][keep].argmax()
        score = output["scores"][keep][best_idx].item()
        soft_mask = output["masks"][keep][best_idx, 0].cpu().numpy()
        binary_mask = soft_mask >= self.mask_threshold
        box = output["boxes"][keep][best_idx].cpu().numpy()

        return binary_mask, box, score

    def _crop_and_preprocess(self, image: torch.Tensor, box: np.ndarray) -> torch.Tensor | None:
        """
        Crop the lesion region and preprocess it for the classifier.

        x1/y1 are floored and x2/y2 are ceiled to ensure the full predicted box is included in the crop. Returns None
        if the resulting crop has zero height or width after clamping to image bounds.

        Args:
            image: FloatTensor (1, 3, img_size, img_size) ImageNet-normalised, on self.device.
            box: Float ndarray [x1, y1, x2, y2] in pixel coordinates.

        Returns:
            FloatTensor (1, 3, img_size, img_size) ImageNet-normalised, on self.device. None if bounding box is zero
            after clamping.
        """

        H, W = image.shape[1], image.shape[2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(W, int(box[2]))
        y2 = min(H, int(box[3]))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[:, y1:y2, x1:x2]
        crop = F.interpolate(
            crop.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        mean = torch.tensor(self.class_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.class_std, device=self.device).view(1, 3, 1, 1)

        return (crop - mean) / std

    @staticmethod
    def _compute_jaccard(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        """
        Computed threshold Jaccard for a single image pair.

        Per the ISIC 2018 evaluation protocol (Codella et al., 2019), IoU values <= threshold are set to 0.
        Union == 0.0 returns 0.0 (corrupt model).

        Args:
            pred: Binary predicted mask (H, W).
            gt: Binary ground truth mask (H, W).
            threshold: Jaccard threshold T.

        Returns:
            Threshold Jaccard in [0, 1]
        """

        pred = pred.astype(bool)
        gt = gt.astype(bool)
        union = (pred | gt).sum()
        if union == 0:
            return 0.0

        iou = float((pred & gt).sum() / union)
        return iou if iou >= threshold else 0.0

    # --------------------------------------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> PredictionResult:
        """
        Run the full pipeline on a single image.

        Args:
            image: FloatTensor (3, H, W) in [0,1], no ImageNet normalization. Moved to self.device internally if needed.

        Returns:
            PredictionResult with mask, class_label, class_probabilities, detection_failed and score fields.
        """
        image = image.to(self.device)

        binary_mask, box, score = self._run_segmentation(image)

        if box is None:
            return PredictionResult(
                mask=binary_mask, class_label=-1, class_probabilities=None, detection_failed=True, score=0.0
            )

        crop = self._crop_and_preprocess(image, box)
        if crop is None:
            return PredictionResult(
                mask=binary_mask, class_label=-1, class_probabilities=None, detection_failed=True, score=0.0
            )

        logit = self.class_model(crop)
        probs = torch.softmax(logit, dim=1)[0].cpu().numpy()
        label = int(probs.argmax())

        return PredictionResult(
            mask=binary_mask, class_label=label, class_probabilities=probs, detection_failed=False, score=score
        )

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> PredictionResult:
        """
        Evaluate the full pipeline on a validation set.

        The Dataloader must yield batched of (image, gt_mask, gt_class_label):
            image: FloatTensor (1, 3, H, W) in [0, 1].
            gt_mask: BoolTensor or uint8 Tensor (1, H, W).
            gt_class_label: Scalar Tensor or int. Ground truth class label.

        Jaccard is computed over all images, detection failures contribute 0.0.
        Balanced accuracy is computed only over images where detection succeeded.
        The combined score is the arithmetic mean of mean_jaccard and balanced_accuracy. If no image produces a valid
        detection, balanced_accuracy and combined_score are both None.

        Args:
            dataloader: DataLoader yielding (image, gt_mask, gt_class_label). Must use batch_size=1.

        Returns:
            EvaluationResult with mean_jaccard, balanced_accuracy, combined_score, n_total and n_detection_failed.
        """

        self.segment_model.eval()
        self.class_model.eval()

        jaccard_scores: list[float] = []
        class_preds: list[int] = []
        class_labels: list[int] = []
        n_detection_failed = 0

        for batch in dataloader:
            image, gt_mask, gt_class_label = batch

            if image.shape[0] != 1:
                raise ValueError(
                    f"evaluate() requires batch_size=1, got {image.shape[0]}." "predict() processes one image at a time"
                )

            image = image[0]
            gt_mask = gt_mask[0].numpy().astype(bool)
            gt_label = int(gt_class_label.item()) if isinstance(gt_class_label, torch.Tensor) else int(gt_class_label)

            result = self.predict(image)

            jaccard_scores.append(self._compute_jaccard(result.mask, gt_mask, self.jaccard_threshold))

            if result.detection_failed:
                n_detection_failed += 1
            else:
                class_preds.append(result.class_label)
                class_labels.append(gt_label)

        n_total = len(jaccard_scores)
        mean_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0

        if class_preds:
            bal_acc = float(balanced_accuracy_score(class_labels, class_preds))
            combined = (mean_jaccard + bal_acc) / 2
        else:
            bal_acc, combined = None, None

        return EvaluationResult(
            mean_jaccard=mean_jaccard,
            balanced_accuracy=bal_acc,
            combined_score=combined,
            n_total=n_total,
            n_detection_failed=n_detection_failed,
        )
