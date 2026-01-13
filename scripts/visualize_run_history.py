import argparse
import datetime
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import autorootcwd

from src.loggings.run_history import load_run_history


def _normalize_timestamp(ts: str) -> datetime.datetime:
    # Parse ISO strings, treat 'Z' as UTC, default to UTC if naive, then convert to KST (UTC+9)
    s = (ts or "").strip()
    if not s:
        # Return a deterministic minimal time in KST
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc).astimezone(
            datetime.timezone(datetime.timedelta(hours=9))
        )

    # Handle trailing 'Z' as UTC explicitly
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")

    dt = datetime.datetime.fromisoformat(s)
    if dt.tzinfo is None:
        # Assume UTC for naive timestamps
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    # Convert to Korea Standard Time (UTC+9)
    kst = datetime.timezone(datetime.timedelta(hours=9))
    return dt.astimezone(kst)


def _summarize(history: List[Dict[str, Any]]):
    by_mode_type = Counter()
    latest_by_mode: Dict[str, Dict[str, Any]] = {}
    for item in history:
        mode = item.get("mode", "unknown")
        diffusion_type = item.get("diffusion_type", "unknown")
        by_mode_type[(mode, diffusion_type)] += 1
        latest_by_mode[mode] = item
    return by_mode_type, latest_by_mode


def _render_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    buf = io.StringIO()
    buf.write("| " + " | ".join(headers) + " |\n")
    buf.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for r in rows:
        buf.write("| " + " | ".join(str(x) for x in r) + " |\n")
    return buf.getvalue()


def save_tables(history: List[Dict[str, Any]], output_path: Path) -> None:
    if not history:
        print("No history found; nothing to save.")
        return

    by_mode_type, _ = _summarize(history)
    diffusion_types = sorted({k[1] for k in by_mode_type.keys()})
    modes = sorted({k[0] for k in by_mode_type.keys()})

    # Table 1: Counts by diffusion type per mode
    headers_counts = ["Mode"] + diffusion_types + ["Total"]
    rows_counts: List[List[Any]] = []
    for mode in modes:
        counts = [by_mode_type.get((mode, dt), 0) for dt in diffusion_types]
        rows_counts.append([mode] + counts + [sum(counts)])

    # Grand total row
    grand_counts = [sum(by_mode_type.get((m, dt), 0) for m in modes) for dt in diffusion_types]
    rows_counts.append(["Total"] + grand_counts + [sum(grand_counts)])

    # Table 2: Timeline details
    # Keep original order (latest likely at bottom). If needed, sort by timestamp
    def safe_ts(item: Dict[str, Any]) -> str:
        try:
            # Format timestamp as 연/월/일 시:분:초
            return _normalize_timestamp(item.get("timestamp", "")).strftime("%Y/%m/%d %H:%M:%S")
        except Exception:
            return str(item.get("timestamp", ""))

    headers_timeline = [
        "#",
        "Timestamp (KST)",
        "Mode",
        "Diffusion",
        "Dataset",
        "Prior",
        "Epoch",
    ]
    rows_timeline: List[List[Any]] = []
    for idx, item in enumerate(history, start=1):
        rows_timeline.append([
            idx,
            safe_ts(item),
            item.get("mode", "unknown"),
            item.get("diffusion_type", "unknown"),
            item.get("dataset", "-"),
            item.get("prior_model", "-"),
            item.get("epoch", "-"),
        ])

    # Table 3: Compact per-run details
    def fmt_bool(v: Any) -> str:
        return "Y" if bool(v) else "N" if v is not None else "-"

    headers_details = [
        "#",
        "Timestamp (KST)",
        "Mode",
        "PID",
        "Diffusion",
        "Dataset",
        "Prior",
        "Epoch",
        "Batch",
        "LR",
        "FP16",
        "DDIM",
        "Image",
        "Sampler",
        "GPU",
        "Resume",
    ]
    rows_details: List[List[Any]] = []
    for idx, item in enumerate(history, start=1):
        resume_ckpt = item.get("resume_checkpoint") or "-"
        rows_details.append([
            idx,
            safe_ts(item),
            item.get("mode", "unknown"),
            item.get("pid", "-"),
            item.get("diffusion_type", "unknown"),
            item.get("dataset", "-"),
            item.get("prior_model", "-"),
            item.get("epoch", "-"),
            item.get("batch_size", "-"),
            item.get("lr", "-"),
            fmt_bool(item.get("use_fp16")),
            fmt_bool(item.get("use_ddim")),
            item.get("image_size", "-"),
            item.get("schedule_sampler", "-"),
            item.get("gpu", "-"),
            resume_ckpt,
        ])

    md = io.StringIO()
    md.write("# Run History Summary\n\n")
    md.write("## Counts by Diffusion Type and Mode\n\n")
    md.write(_render_markdown_table(headers_counts, rows_counts))
    md.write("\n## Timeline\n\n")
    md.write(_render_markdown_table(headers_timeline, rows_timeline))
    md.write("\n## Run Details (Compact)\n\n")
    md.write(_render_markdown_table(headers_details, rows_details))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # If the user passed an image extension, still write markdown alongside
    if output_path.suffix.lower() not in {".md", ".markdown"}:
        # Replace extension with .md
        output_path = output_path.with_suffix(".md")

    output_path.write_text(md.getvalue(), encoding="utf-8")
    print(f"Saved tables to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize run history stored in run_history.json")
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("workdir/run_history.json"),
        help="Path to run_history.json (default: workdir/run_history.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workdir/run_history.md"),
        help="Where to save the table (markdown). If a non-markdown extension is provided, it will be saved as .md next to it. (default: workdir/run_history.md)",
    )
    parser.add_argument(
        "--reset-history",
        action="store_true",
        help="After generating the summary, clear the history file at --history-path by writing an empty list []",
    )
    parser.add_argument(
        "--save-cleared-to",
        type=Path,
        default=None,
        help="If provided with --reset-history, save the current history (JSON) to this path before clearing --history-path",
    )
    args = parser.parse_args()

    history = load_run_history(args.history_path)
    save_tables(history, args.output)

    # Optionally archive and clear the history JSON
    if getattr(args, "reset_history", False):
        try:
            if getattr(args, "save_cleared_to", None):
                args.save_cleared_to.parent.mkdir(parents=True, exist_ok=True)
                args.save_cleared_to.write_text(
                    json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"Archived current history to {args.save_cleared_to}")

            # Clear original history file
            args.history_path.parent.mkdir(parents=True, exist_ok=True)
            args.history_path.write_text("[]", encoding="utf-8")
            print(f"Cleared history at {args.history_path}")
        except Exception as e:
            print(f"Failed to reset history: {e}")


if __name__ == "__main__":
    main()
