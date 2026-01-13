"""Evaluation helpers shared across scripts."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree
from skimage.measure import euler_number, label

logger = logging.getLogger(__name__)

METRICS = [
    ("f1", True),
    ("iou", True),
    ("precision", True),
    ("recall", True),
    ("boundary_ap", True),
    ("boundary_acc", True),
    ("hd95", False),
    ("betti_0_error", False),
    ("betti_1_error", False),
]

DEFAULT_MODELS = ["Unet", "SwinUNETR", "CSNet", "FRUnet"]

PREDICTION_PATTERNS = {
    "final": "final_output",
}
FIGURES_DIR = Path("figures")
WORKDIR_DIR = Path("workdir")
RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
TEX_DIR = RESULTS_DIR / "tex"


def _get_workdir_path(dataset: str, model: str, sampling_method: str = "") -> Path:
    """Create the expected workdir path for a given config."""
    if sampling_method:
        dir_name = f"{sampling_method}-mse-concat/{dataset}-{model}"
    else:
        dir_name = f"{dataset}-{model}"
    return WORKDIR_DIR / dir_name


def boundary_stats(pred_mask: np.ndarray, gt_mask: np.ndarray, max_tol: int = 5) -> Tuple[float, float]:
    """Compute boundary precision/recall statistics."""
    pred_b = np.logical_xor(pred_mask, binary_erosion(pred_mask))
    gt_b = np.logical_xor(gt_mask, binary_erosion(gt_mask))

    if gt_b.sum() == 0 and pred_b.sum() == 0:
        return 1.0, 1.0

    dt_gt = distance_transform_edt(~gt_b)
    dt_pred = distance_transform_edt(~pred_b)

    precisions = []
    recalls = []

    for tol in range(max_tol + 1):
        matches_pred = (dt_gt[pred_b] <= tol).sum() if pred_b.sum() > 0 else 0
        matches_gt = (dt_pred[gt_b] <= tol).sum() if gt_b.sum() > 0 else 0

        prec = matches_pred / pred_b.sum() if pred_b.sum() > 0 else 1.0
        rec = matches_gt / gt_b.sum() if gt_b.sum() > 0 else 1.0

        precisions.append(prec)
        recalls.append(rec)

    boundary_ap = float(np.mean(precisions))
    boundary_acc = float(recalls[min(1, len(recalls) - 1)])
    return boundary_ap, boundary_acc


def compute_betti_numbers(mask: np.ndarray, connectivity: int = 1) -> Tuple[int, int]:
    """Return Betti-0 and Betti-1 counts for a binary mask."""
    labeled = label(mask, connectivity=connectivity)
    beta0 = labeled.max()

    euler = euler_number(mask, connectivity=connectivity)
    beta1 = beta0 - euler

    return beta0, beta1


def betti_error(pred_mask: np.ndarray, gt_mask: np.ndarray, connectivity: int = 1) -> Tuple[int, int, int]:
    """Compute the Betti error between prediction and ground truth."""
    p0, p1 = compute_betti_numbers(pred_mask, connectivity)
    g0, g1 = compute_betti_numbers(gt_mask, connectivity)

    error0 = abs(p0 - g0)
    error1 = abs(p1 - g1)
    total_error = error0 + error1

    return total_error, error0, error1


def calculate_metrics(label: Union[np.ndarray, object], pred: Union[np.ndarray, object]) -> Dict[str, float]:
    """Compute per-image segmentation metrics."""
    if hasattr(label, "cpu"):
        label = label.cpu().numpy()
    if hasattr(pred, "cpu"):
        pred = pred.cpu().numpy()

    label = (label - label.min()) / (label.max() - label.min() + 1e-8)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    label = (label > 0.5).astype(np.bool_)
    pred = (pred > 0.5).astype(np.bool_)

    intersection = np.logical_and(label, pred).sum()
    union = np.logical_or(label, pred).sum()

    tp = intersection
    fp = pred.sum() - tp
    fn = label.sum() - tp

    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    iou = intersection / (union + 1e-8) if union > 0 else 1.0

    b_ap, b_acc = boundary_stats(pred, label, max_tol=5)

    label_points = np.argwhere(label)
    pred_points = np.argwhere(pred)
    if label_points.size == 0 or pred_points.size == 0:
        hd95 = float("inf")
    else:
        pred_tree = cKDTree(pred_points)
        label_tree = cKDTree(label_points)

        min_dists_label, _ = pred_tree.query(label_points, k=1)
        min_dists_pred, _ = label_tree.query(pred_points, k=1)
        hd95 = np.percentile(np.concatenate([min_dists_label, min_dists_pred]), 95)

    _, betti0_error, betti1_error = betti_error(pred, label)

    return {
        "f1": float(f1),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "boundary_ap": b_ap,
        "boundary_acc": b_acc,
        "hd95": float(hd95),
        "betti_0_error": float(betti0_error),
        "betti_1_error": float(betti1_error),
    }


def evaluate_results(
    dataset: str,
    model: str,
    epoch: int = 50000,
    sampling_method: str = "poisson",
    prediction_type: str = "final",
):
    """Evaluate a single configuration and cache metrics CSV."""
    work_path = _get_workdir_path(dataset, model, sampling_method)
    result_path = work_path / f"results-{epoch:06d}"

    if not result_path.exists():
        logger.warning(f"Results path not found: {result_path}")
        return

    if prediction_type == "final":
        outputs = natsorted([p for p in result_path.glob("*final_output*.npy")])
    elif prediction_type == "input":
        outputs = natsorted([p for p in result_path.glob("*input*.npy")])
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    if not outputs:
        logger.warning(f"No output files found for type '{prediction_type}' in: {result_path}")
        return
    targets = natsorted([p for p in result_path.glob("*target*.npy")])
    img_postfix = outputs[0].suffix.lower()
    tgt_postfix = targets[0].suffix.lower()

    if len(outputs) != len(targets):
        logger.warning(f"Warning: Mismatched between {len(outputs)} outputs and {len(targets)} targets at {result_path}")
        # Create a CSV mapping outputs -> targets for quick inspection
        min_len = min(len(outputs), len(targets))
        pairs = [(str(outputs[i]), str(targets[i])) for i in range(min_len)]
        df_pairs = pd.DataFrame(pairs, columns=["outputs", "targets"])
        suffix = "" if prediction_type == "final" else f"-{prediction_type}"
        out_pairs_csv = work_path / f"{epoch:06d}-pairs{suffix}.csv"
        df_pairs.to_csv(out_pairs_csv, index=False)
        logger.info(f"Saved pairs CSV: {out_pairs_csv}")

        # Also save full lists so excess items are visible
        df_outs = pd.DataFrame([str(p) for p in outputs], columns=["outputs"])
        df_tgts = pd.DataFrame([str(p) for p in targets], columns=["targets"])
        out_outs_csv = work_path / f"{epoch:06d}-outputs{suffix}.csv"
        out_tgts_csv = work_path / f"{epoch:06d}-targets{suffix}.csv"
        df_outs.to_csv(out_outs_csv, index=False)
        df_tgts.to_csv(out_tgts_csv, index=False)
        logger.info(f"Saved outputs list CSV: {out_outs_csv}")
        logger.info(f"Saved targets list CSV: {out_tgts_csv}")

    if not outputs:
        return

    rows = []
    for i, (out_p, tgt_p) in enumerate(zip(outputs, targets)):
        if img_postfix == '.npy':
            out_arr = np.load(out_p, allow_pickle=True)
        else:
            out_arr = np.array(Image.open(out_p).convert("L")).astype(np.uint8)
        if tgt_postfix == '.npy':
            tgt_arr = np.load(tgt_p, allow_pickle=True)
        else:
            tgt_arr = np.array(Image.open(tgt_p).convert("L")).astype(np.uint8)

        out_denom = (out_arr.max() - out_arr.min())
        out_arr = (out_arr - out_arr.min()) / (out_denom + 1e-8)
        out_arr = (out_arr > 0.5).astype(np.uint8)
        tgt_denom = (tgt_arr.max() - tgt_arr.min())
        tgt_arr = (tgt_arr - tgt_arr.min()) / (tgt_denom + 1e-8)
        tgt_arr = (tgt_arr > 0.5).astype(np.uint8)

        out_arr = np.squeeze(out_arr)
        tgt_arr = np.squeeze(tgt_arr)

        m = calculate_metrics(tgt_arr, out_arr)
        m.update({"file": out_p.name, "batch": f"batch_{i}"})
        rows.append(m)

    df = pd.DataFrame(rows)
    metric_cols = [col for col in df.columns if col not in ["file", "batch"]]
    stats = df[metric_cols].agg(["mean", "std"]).T

    overall_data = {"file": "overall", "batch": "overall"}
    for idx, row in stats.iterrows():
        overall_data[idx] = f"{row['mean']:.4f}±{row['std']:.4f}"

    df_final = pd.concat([pd.DataFrame([overall_data]), df], ignore_index=True)

    suffix = "" if prediction_type == "final" else f"-{prediction_type}"
    # 통일된 metrics 경로: results/metrics/{sampling_method}/{dataset}-{model}/metrics-{epoch:06d}{suffix}.csv
    metrics_csv_path = METRICS_DIR / sampling_method / f"{dataset}-{model}" / f"metrics-{epoch:06d}{suffix}.csv"
    os.makedirs(metrics_csv_path.parent, exist_ok=True)
    df_final.to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved metrics to: {metrics_csv_path}")


def load_model_metrics(
    dataset: str,
    model: str,
    sampling_method: str = "",
    prediction_type: str = "final",
    epoch: int = 50000,
) -> Optional[Dict[str, Union[str, float]]]:
    """Load averaged metrics row from CSV if present."""
    suffix = "" if prediction_type == "final" else f"-{prediction_type}"
    # 통일된 metrics 경로: results/metrics/{sampling_method}/{dataset}-{model}/metrics-{epoch:06d}{suffix}.csv
    metrics_csv_path = METRICS_DIR / sampling_method / f"{dataset}-{model}" / f"metrics-{epoch:06d}{suffix}.csv"
    if metrics_csv_path.exists():
        df = pd.read_csv(metrics_csv_path)
        if "file" in df.columns:
            df.rename(columns={"file": "file_name"}, inplace=True)

        target = df[df["file_name"].isin(["avg±std_metrics", "overall"])]
        if not target.empty:
            row = target.iloc[0].to_dict()
            if row["file_name"] == "overall":
                row["file_name"] = "avg±std_metrics"
            return row
    return None


__all__ = [
    "RESULTS_DIR",
    "WORKDIR_DIR",
    "METRICS_DIR",
    "METRICS",
    "DEFAULT_MODELS",
    "PREDICTION_PATTERNS",
    "_get_workdir_path",
    "evaluate_results",
    "load_model_metrics",
    "calculate_metrics",
]
