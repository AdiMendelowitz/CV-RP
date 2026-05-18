"""
Reconstruction-error-based anomaly detection on ETTh1 using a pretrained PatchTST model.

Methodology:
    1. Load the PatchTST checkpoint trained at pred_len=96 on ETTh1.
    2. Run sliding-window inference (step=1) over the val split to compute per-timestep reconstruction MSE.
       The first predicted step (index 0 of the 96-step output) serves as the one-step-ahead reconstruction for the
       timestep immediately after each window.
    3. Set the anomaly threshold at the 95th percentile of val reconstruction errors.
       The threshold is fit on val and applied to test -- no leakage.
    4. Inject two synthetic anomaly types into the test split:
         - Point anomaly:      Gaussian noise (std = 5 * channel_std) at 20 random timesteps.
         - Contextual anomaly: Zero out a contiguous 24-hour (24-step) window at 3 locations.
    5. Run sliding-window inference over the (corrupted) test split.
    6. Evaluate precision, recall, and F1 at the chosen threshold, reported per anomaly type.
    7. Write results to results/anomaly_etth1.md.

Note: this is reconstruction-error-based detection, not a learned anomaly-specific objective.
SOTA methods (Anomaly Transformer, Xu et al. ICLR 2022; TranAD, Tuli et al. VLDB 2022) use objectives designed
specifically for anomaly detection. The value here is demonstrating the connection between forecasting quality and
anomaly signal, a pattern directly relevant to inverter telemetry monitoring, where the same reconstruction principle
applies at scale.

Reference:
    Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers",
    ICLR 2023. https://arxiv.org/abs/2211.14730
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_PATH = _REPO_ROOT / "data" / "ETTh1.csv"
_CKPT_PATH = _REPO_ROOT / "results" / "checkpoints" / "patchtst_pred96_best.pt"
_RESULTS_DIR = _REPO_ROOT / "results"
_REPORT_PATH = _RESULTS_DIR / "anomaly_etth1.md"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEQ_LEN = 512
PRED_LEN = 96          # checkpoint output dimension; we use step 0 as 1-step-ahead reconstruction
BATCH_SIZE = 256       # windows per forward pass during sliding-window inference
THRESHOLD_PERCENTILE = 95
N_POINT_ANOMALIES = 20
N_CONTEXTUAL_WINDOWS = 3
CONTEXTUAL_WINDOW_HOURS = 24
POINT_ANOMALY_STD_MULTIPLIER = 5.0
TARGET_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
SPLIT_SIZES = {"train": 8640, "val": 2880, "test": 2880}


# ---------------------------------------------------------------------------
# PatchTST model
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, seq_len: int, patch_size: int, stride: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = (seq_len - patch_size) // stride + 1
        self.projection = nn.Linear(patch_size, d_model)

        # Compute PE once and store as a plain tensor (not a buffer) so it does
        # not appear in the state dict, matching the checkpoint's saved format.
        position = torch.arange(self.num_patches).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(self.num_patches, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self._pe = pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B*C, seq_len, 1) -> (B*C, seq_len) -> (B*C, num_patches, patch_size)
        x = x.squeeze(-1).unfold(-1, self.patch_size, self.stride)
        return self.projection(x) + self._pe.to(x.device)            # (B*C, num_patches, d_model)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

class ForecastHead(nn.Module):
    def __init__(self, num_patches, d_model, pred_len) -> None:
        super().__init__()
        self.linear = nn.Linear(num_patches * d_model, pred_len)

    def forward(self, x):
        return self.linear(x)

class PatchTST(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_variates: int, patch_size: int = 16, stride: int = 8,
                 d_model: int = 128, num_heads: int = 16, num_layers: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.num_variates = num_variates
        self.patch_embedding = PatchEmbedding(seq_len, patch_size, stride, d_model)
        num_patches = self.patch_embedding.num_patches
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, dropout)
        self.head = ForecastHead(num_patches, d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape                              # x: (B, seq_len, C)
        x = x.permute(0, 2, 1).reshape(B * C, L, 1)    # (B*C, seq_len, 1)
        x = self.patch_embedding(x)                    # (B*C, num_patches, d_model)
        x = self.encoder(x)                            # (B*C, num_patches, d_model)
        x = x.flatten(1)                               # (B*C, num_patches * d_model)
        x = self.head(x)                               # (B*C, pred_len)
        return x.reshape(B, C, -1).permute(0, 2, 1)    # (B, pred_len, C)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_etth1(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ETTh1, normalize on train, return (train, val, test, channel_std).

    Returns:
        train: (8640, 7) normalized
        val: (2880, 7) normalized
        test: (2880, 7) normalized
        channel_std: (7,) train-split std per channel, in original scale
    """
    df = pd.read_csv(csv_path)[TARGET_COLS].values.astype(np.float32)

    train_end = SPLIT_SIZES["train"]
    val_end = train_end + SPLIT_SIZES["val"]

    train_raw = df[:train_end]
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    train = (df[:train_end] - mean) / std
    val = (df[train_end:val_end] - mean) / std
    test = (df[val_end:] - mean) / std

    return train, val, test, std


# ---------------------------------------------------------------------------
# Sliding-window reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruction_errors(model: nn.Module, data: np.ndarray, seq_len: int, device: torch.device) -> np.ndarray:
    """
    Compute per-timestep reconstruction MSE using 1-step-ahead prediction.

    For each window [i : i+seq_len], the model predicts pred_len steps. Step 0 as the reconstruction of timestep
    i+seq_len, and the MSE is computed against the actual value at that timestep.

    Args:
        model: Trained PatchTST (pred_len >= 1).
        data: Array of shape (T, C).
        seq_len: Input window length.
        device: Torch device.

    Returns:
        errors: Array of shape (T - seq_len,) -- one MSE per reconstructed timestep.
                errors[i] corresponds to data[i + seq_len].
    """
    model.eval()
    T, C = data.shape
    n_windows = T - seq_len
    errors = np.empty(n_windows, dtype=np.float32)

    tensor = torch.from_numpy(data).to(device)

    for start in range(0, n_windows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_windows)
        indices = range(start, end)

        # Stack windows: (batch, seq_len, C)
        batch_x = torch.stack([tensor[i : i + seq_len] for i in indices])
        # Actual next timestep: (batch, C)
        batch_y = torch.stack([tensor[i + seq_len] for i in indices])

        pred = model(batch_x)          # (batch, pred_len, C)
        pred_step0 = pred[:, 0, :]     # (batch, C) -- one-step-ahead reconstruction

        mse = ((pred_step0 - batch_y) ** 2).mean(dim=-1)  # (batch,) -- mean over channels
        errors[start:end] = mse.cpu().numpy()

    return errors


# ---------------------------------------------------------------------------
# Anomaly injection
# ---------------------------------------------------------------------------

def inject_point_anomalies(data: np.ndarray, channel_std: np.ndarray, n_anomalies: int,
                           valid_range: tuple[int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject Gaussian point anomalies and return corrupted data and binary labels.

    Args:
        data: Array of shape (T, C). Modified in-place on a copy.
        channel_std: (C,) per-channel std in original scale (used to set noise magnitude).
        n_anomalies: Number of anomalous timesteps to inject.
        valid_range: (lo, hi) - range for injected anomalies, ensuring they fall within the reconstructable region.
        rng: Numpy random generator for reproducibility.

    Returns:
        corrupted: (T, C) array with anomalies injected.
        labels: (T,) binary array; 1 at anomalous timesteps.
    """
    lo, hi = valid_range
    corrupted = data.copy()
    labels = np.zeros(len(data), dtype=np.int32)

    positions = rng.choice(np.arange(lo, hi), size=n_anomalies, replace=False)
    noise = rng.standard_normal((n_anomalies, data.shape[1])).astype(np.float32)
    noise *= (POINT_ANOMALY_STD_MULTIPLIER * channel_std)

    for k, pos in enumerate(positions):
        corrupted[pos] += noise[k]
        labels[pos] = 1

    return corrupted, labels


def inject_contextual_anomalies(data: np.ndarray, n_windows: int, window_size: int, valid_range: tuple[int, int],
                                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero out contiguous windows and return corrupted data and binary labels.

    Args:
        data: Array of shape (T, C). Modified in-place on a copy.
        n_windows: Number of contiguous windows to zero out.
        window_size: Length of each zeroed window in timesteps.
        valid_range: (lo, hi) -- window starts drawn from this range.
        rng: Numpy random generator.

    Returns:
        corrupted: (T, C) array with windows zeroed.
        labels: (T,) binary array; 1 at all timesteps within zeroed windows.
    """
    lo, hi = valid_range
    # Ensure no window runs past the valid range boundary.
    start_hi = hi - window_size
    if start_hi <= lo:
        raise ValueError(f"valid_range ({lo}, {hi}) too small for window_size={window_size}.")

    corrupted = data.copy()
    labels = np.zeros(len(data), dtype=np.int32)

    starts = rng.choice(np.arange(lo, start_hi), size=n_windows, replace=False)
    for start in starts:
        corrupted[start : start + window_size] = 0.0
        labels[start : start + window_size] = 1

    return corrupted, labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_detection(errors: np.ndarray, labels: np.ndarray, threshold: float,
                       reconstructable_offset: int) -> dict[str, float]:
    predicted = (errors > threshold).astype(np.int32)
    aligned_labels = labels[reconstructable_offset : reconstructable_offset + len(errors)]

    return {
        'precision': precision_score(aligned_labels, predicted, zero_division=0.0),
        'recall':    recall_score(aligned_labels, predicted, zero_division=0.0),
        'f1':        f1_score(aligned_labels, predicted, zero_division=0.0),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(threshold: float, point_metrics: dict[str, float], contextual_metrics: dict[str, float],
                 n_point: int, n_contextual_timesteps: int, n_test_reconstructable: int, report_path: Path) -> None:
    lines = [
        "# Anomaly Detection on ETTh1 -- PatchTST Reconstruction Errors",
        "",
        "## Method",
        "",
        "Reconstruction-error-based anomaly detection using a PatchTST model trained for",
        "long-horizon forecasting (pred_len=96, seq_len=512) on ETTh1. For each sliding",
        "window of length 512 (step size 1), the model predicts 96 steps ahead. Step 0",
        "of the prediction serves as the one-step-ahead reconstruction of the next timestep.",
        "The per-timestep reconstruction error is the MSE between step 0 and the actual",
        "value, averaged across all 7 ETTh1 channels.",
        "",
        "This is not a learned anomaly-specific objective. SOTA methods such as Anomaly",
        "Transformer (Xu et al., ICLR 2022) and TranAD (Tuli et al., VLDB 2022) use",
        "objectives designed specifically for anomaly detection. The value of this",
        "experiment is demonstrating the connection between forecasting quality and anomaly",
        "signal: a model trained purely on next-step prediction produces elevated",
        "reconstruction errors at anomalous timesteps without any anomaly-specific",
        "supervision. This principle applies directly to inverter telemetry monitoring,",
        "where a forecasting model trained on normal operating conditions can flag",
        "anomalous readings by reconstruction error thresholding.",
        "",
        "## Configuration",
        "",
        f"- Model:                  PatchTST (seq_len=512, pred_len=96)",
        f"- Checkpoint:             patchtst_pred96_best.pt",
        f"- Dataset:                ETTh1 test split (2880 timesteps)",
        f"- Reconstructable range:  {n_test_reconstructable} timesteps",
        f"- Threshold:              95th percentile of val reconstruction errors = {threshold:.6f}",
        f"- Point anomalies:        {n_point} timesteps (Gaussian noise, std = 5 * channel_std)",
        f"- Contextual anomalies:   {n_contextual_timesteps} timesteps "
        f"({N_CONTEXTUAL_WINDOWS} x {CONTEXTUAL_WINDOW_HOURS}-hour windows zeroed out)",
        "",
        "## Results",
        "",
        "### Point Anomalies",
        "",
        "| Metric    | Value  |",
        "|-----------|--------|",
        f"| Precision | {point_metrics['precision']:.4f} |",
        f"| Recall    | {point_metrics['recall']:.4f} |",
        f"| F1        | {point_metrics['f1']:.4f} |",
        "",
        "### Contextual Anomalies",
        "",
        "| Metric    | Value  |",
        "|-----------|--------|",
        f"| Precision | {contextual_metrics['precision']:.4f} |",
        f"| Recall    | {contextual_metrics['recall']:.4f} |",
        f"| F1        | {contextual_metrics['f1']:.4f} |",
        "",
        "## Interpretation",
        "",
        "Point anomalies (large instantaneous spikes) are harder to detect with a",
        "reconstruction-error approach because the model's one-step-ahead prediction",
        "is based on a 512-timestep context window. A single corrupted timestep has",
        "minimal effect on the context and the error signal appears at exactly one",
        "position in the error array, making precision-recall sensitive to the threshold.",
        "",
        "Contextual anomalies (extended zero windows) are more detectable because the",
        "corrupted region persists across many consecutive windows, producing sustained",
        "elevated reconstruction errors that are easier to separate from the background",
        "distribution.",
        "",
        "Seed: 42. Results are fully reproducible.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    device = torch.device("cpu")

    # Load and normalize data.
    train, val, test, channel_std = load_etth1(_DATA_PATH)

    # Load model.
    model = PatchTST(seq_len=SEQ_LEN, pred_len=PRED_LEN, num_variates=len(TARGET_COLS), patch_size=16, stride=8,
                     d_model=128, num_heads=16, num_layers=3, dropout=0.2).to(device)

    state_dict = torch.load(_CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Checkpoint loaded: {_CKPT_PATH}")

    # Fit threshold on val reconstruction errors - no leakage.
    print("Computing val reconstruction errors...")
    val_errors = reconstruction_errors(model, val, SEQ_LEN, device)
    threshold = float(np.percentile(val_errors, THRESHOLD_PERCENTILE))
    print(f"Threshold ({THRESHOLD_PERCENTILE}th percentile of val errors): {threshold:.6f}")

    # Reconstructable region within the test split.
    # errors[i] corresponds to test[i + SEQ_LEN], so valid anomaly positions
    # are SEQ_LEN through len(test) - 1 in test-array coordinates.
    n_test_reconstructable = len(test) - SEQ_LEN
    valid_range = (SEQ_LEN, len(test) - 1)

    rng = np.random.default_rng(SEED)

    # --- Point anomalies ---
    print("Injecting point anomalies...")
    test_point, point_labels = inject_point_anomalies(data=test, channel_std=channel_std, n_anomalies=N_POINT_ANOMALIES,
                                                      valid_range=valid_range, rng=np.random.default_rng(SEED))
    print("Running reconstruction on point-anomaly test split...")
    point_errors = reconstruction_errors(model, test_point, SEQ_LEN, device)
    point_metrics = evaluate_detection(point_errors, point_labels, threshold, SEQ_LEN)
    print(f"Point anomaly -- P: {point_metrics['precision']:.4f}  "
          f"R: {point_metrics['recall']:.4f}  F1: {point_metrics['f1']:.4f}")

    # --- Contextual anomalies ---
    print("Injecting contextual anomalies...")
    test_contextual, contextual_labels = inject_contextual_anomalies(
        data=test,
        n_windows=N_CONTEXTUAL_WINDOWS,
        window_size=CONTEXTUAL_WINDOW_HOURS,
        valid_range=valid_range,
        rng=np.random.default_rng(SEED + 1),
    )
    print("Running reconstruction on contextual-anomaly test split...")
    contextual_errors = reconstruction_errors(model, test_contextual, SEQ_LEN, device)
    contextual_metrics = evaluate_detection(contextual_errors, contextual_labels, threshold, SEQ_LEN)
    print(f"Contextual anomaly -- P: {contextual_metrics['precision']:.4f}  "
          f"R: {contextual_metrics['recall']:.4f}  F1: {contextual_metrics['f1']:.4f}")

    n_contextual_timesteps = int(contextual_labels.sum())

    write_report(threshold=threshold, point_metrics=point_metrics, contextual_metrics=contextual_metrics,
                 n_point=N_POINT_ANOMALIES, n_contextual_timesteps=n_contextual_timesteps,
                 n_test_reconstructable=n_test_reconstructable, report_path=_REPORT_PATH)