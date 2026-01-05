import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import autorootcwd


def _read_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        # Fall back to empty list if file is corrupted
        return []
    return []


def _pluck_args(args: Any, keys: Iterable[str]) -> Dict[str, Any]:
    return {k: getattr(args, k) for k in keys if hasattr(args, k)}


def append_run_history(args: Any, mode: str, command: str, **extra: Any) -> None:
    """
    Append a run record into base_output/run_history.json using values from args.

    Args:
        args: Parsed argparse Namespace or any object with attributes.
        mode: "train" or "sample".
        command: Entry-point name (e.g., "image_train").
        extra: Any additional key-value pairs to store (e.g., output_dir, result_dir).
    """

    base_output = "test-run" if getattr(args, "test_run", False) else "workdir"
    history_path = Path(base_output) / "run_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    common_keys = [
        "diffusion_type",
        "dataset",
        "prior_model",
        "model_path",
        "resume_checkpoint",
        "schedule_sampler",
        "batch_size",
        "lr",
        "ema_rate",
        "use_fp16",
        "use_ddim",
        "stride",
        "gpu",
        "image_size",
        "class_cond",
        "test_run",
    ]

    record: Dict[str, Any] = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "mode": mode,
        "command": command,
        "pid": os.getpid(),
        **_pluck_args(args, common_keys),
        **extra,
    }

    # Drop None values to keep JSON clean
    record = {k: v for k, v in record.items() if v is not None}

    history = _read_history(history_path)
    history.append(record)

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_run_history(history_path: Path) -> List[Dict[str, Any]]:
    """Load history from the given path. Returns an empty list when missing."""
    return _read_history(history_path)
