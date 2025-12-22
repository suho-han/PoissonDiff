"""Utility script to inspect Betti number discrepancies for segmentation masks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from natsort import natsorted
from PIL import Image

from src.scripts.create_table import _get_workdir_path, compute_betti_numbers


def _load_pairs(result_path: Path) -> Iterable[Tuple[Path, Path]]:
    outputs = natsorted(result_path.glob("*final_output*"))
    targets = natsorted(result_path.glob("*target*"))
    return zip(outputs, targets)


def _to_binary(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr > 127


def analyze_betti_errors(dataset: str, model: str, sampling_method: str, epoch: int, limit: int) -> None:
    work_path = _get_workdir_path(dataset, model, sampling_method)
    result_path = work_path / f"results-{epoch:06d}"

    if not result_path.exists():
        print(f"Result path not found: {result_path}")
        return

    total = 0
    mismatches = 0

    for idx, (out_p, tgt_p) in enumerate(_load_pairs(result_path)):
        if limit and idx >= limit:
            break

        pred_arr = np.array(Image.open(out_p))
        label_arr = np.array(Image.open(tgt_p))

        pred_mask = _to_binary(pred_arr)
        label_mask = _to_binary(label_arr)

        b0_pred, b1_pred = compute_betti_numbers(pred_mask.astype(int))
        b0_label, b1_label = compute_betti_numbers(label_mask.astype(int))

        diff_b0 = abs(b0_pred - b0_label)
        diff_b1 = abs(b1_pred - b1_label)

        if diff_b0 > 0 or diff_b1 > 0:
            mismatches += 1

        print(f"[{idx}] {out_p.name}")
        print(
            f"    mass: pred={pred_mask.sum()} label={label_mask.sum()} | "
            f"betti0: pred={b0_pred} label={b0_label} diff={diff_b0} | "
            f"betti1: pred={b1_pred} label={b1_label} diff={diff_b1}"
        )

        total += 1

    print(f"Checked {total} samples, mismatches: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Betti number differences for stored predictions.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., DRIVE)")
    parser.add_argument("--model", required=True, help="Model name (e.g., Unet)")
    parser.add_argument("--diffusion_type", required=True, help="Sampling method identifier")
    parser.add_argument("--epoch", type=int, default=50000, help="Epoch number used in folder naming")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of samples to inspect")

    args = parser.parse_args()
    analyze_betti_errors(args.dataset, args.model, args.diffusion_type, args.epoch, args.limit)


if __name__ == "__main__":
    main()
