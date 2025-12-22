import itertools
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure running from project root if available; ignore if missing
try:
    import autorootcwd  # noqa: F401
except Exception:
    pass
import click

try:
    from tqdm import tqdm
except Exception:  # Fallback if tqdm is unavailable
    def tqdm(iterable, **kwargs):
        return iterable
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from natsort import natsorted
from PIL import Image

from src.scripts.evaluate import DEFAULT_MODELS, FIGURES_DIR, METRICS, METRICS_DIR, PREDICTION_PATTERNS, RESULTS_DIR, TEX_DIR, WORKDIR_DIR, _get_workdir_path, evaluate_results, load_model_metrics

logger = logging.getLogger(__name__)


def _parse_metric_val(val_str: str) -> Tuple[float, float]:
    """Helper to parse 'mean±std' string."""
    try:
        if isinstance(val_str, str) and '±' in val_str:
            mean, std = val_str.split('±')
            return float(mean), float(std)
        return float(val_str), 0.0
    except (ValueError, TypeError):
        # Use NaN to avoid accidentally being treated as best/worst.
        return float("nan"), float("nan")


def write_latex_table(
    dataset: str,
    sampling_method: str = "",
    models: List[str] = DEFAULT_MODELS,
    prediction_type: str = "final",
):
    """Generate LaTeX table for a specific configuration."""
    results = {}
    for model in models:
        res = load_model_metrics(dataset, model, sampling_method, prediction_type=prediction_type)
        if res:
            results[model] = res

    if not results:
        logger.warning(f"No results for {dataset} ({sampling_method}) [{prediction_type}]")
        return

    # Best value 찾기 로직 간소화
    metric_goal = {m[0]: m[1] for m in METRICS}  # True=maximize, False=minimize
    best_vals: Dict[str, Optional[float]] = {m[0]: None for m in METRICS}
    parsed_results = {}  # {model: {metric: (mean, std)}}

    for model, res_dict in results.items():
        parsed_results[model] = {}
        for m_key, _ in METRICS:
            val = res_dict.get(m_key, 0)
            mean, std = _parse_metric_val(val)
            parsed_results[model][m_key] = (mean, std)

            # Skip invalid values when selecting best.
            if np.isnan(mean):
                continue

            maximize = metric_goal[m_key]
            current_best = best_vals[m_key]
            if current_best is None:
                best_vals[m_key] = mean
            elif maximize and mean > current_best:
                best_vals[m_key] = mean
            elif not maximize and mean < current_best:
                best_vals[m_key] = mean

    # LaTeX 작성 (List comprehension 활용)
    lines = [
        "\\documentclass{article}",
        "\\usepackage{booktabs, amssymb, amsmath}",
        "\\usepackage[a4paper,margin=1cm,landscape]{geometry}",
        "\\begin{document}",
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{l" + "c" * len(METRICS) + "}",
        "\\toprule",
        "Model & " + " & ".join([m[0].replace('_', ' ').title() for m in METRICS]) + " \\\\",
        "\\midrule"
    ]

    for model in models:
        if model not in parsed_results:
            continue
        row = [model]
        for m_key, _ in METRICS:
            mean, std = parsed_results[model][m_key]
            if np.isnan(mean) or np.isnan(std):
                row.append("--")
                continue

            val_str = f"{mean:.4f} \\pm {std:.4f}"
            if best_vals[m_key] is not None and np.isclose(mean, best_vals[m_key]):
                val_str = f"\\mathbf{{{val_str}}}"
            row.append(f"${val_str}$")
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Results for {dataset.replace('_', '-')}}}",
        "\\end{table}",
        "\\end{document}"
    ])

    # 파일 저장 및 컴파일
    exp_name_base = f"{sampling_method}_{dataset}" if sampling_method else f"base_{dataset}"
    exp_name = exp_name_base if prediction_type == "final" else f"{exp_name_base}_{prediction_type}"
    tex_path = TEX_DIR / f"sample_{exp_name}.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    _compile_latex(tex_path)


def _compile_latex(tex_path: Path):
    """Compile LaTeX file to PDF."""
    try:
        subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-quiet",
                f"-output-directory={tex_path.parent}",
                str(tex_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return

    pdf_path = tex_path.with_suffix('.pdf')
    if pdf_path.exists():
        shutil.copy(pdf_path, FIGURES_DIR / pdf_path.name)
        print(f"PDF Generated: {FIGURES_DIR / pdf_path.name}")
        # Clean up aux files quietly
        try:
            subprocess.run(
                [
                    "latexmk",
                    "-c",
                    f"-output-directory={tex_path.parent}",
                    str(tex_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def plot_results(dataset: str, sampling_method: str = "", models: List[str] = DEFAULT_MODELS, index: int = 0):
    """Generate comparison plots."""
    model_images = {}

    # 이미지 검색 로직 최적화
    for model in models:
        work_path = _get_workdir_path(dataset, model, sampling_method)
        # 최신 결과 폴더 찾기
        res_dirs = sorted(work_path.glob("results-*"))
        if not res_dirs:
            continue

        latest_dir = res_dirs[-1]
        images = natsorted(list(latest_dir.glob("*image*")))
        inputs = natsorted(list(latest_dir.glob("*input*")))
        targets = natsorted(list(latest_dir.glob("*target*")))
        outputs = natsorted(list(latest_dir.glob("*final_output*")))

        if not (images or inputs or targets or outputs):
            continue

        if index >= len(images) or index >= len(inputs) or index >= len(outputs) or index >= len(targets):
            continue

        # Paths returned by glob() already include latest_dir; don't re-join.
        imgs = {
            "Image": images[index],
            "Input": inputs[index],
            "Output": outputs[index],
            "Target": targets[index],
        }

        if all(p.exists() for p in imgs.values()):
            model_images[model] = imgs

    if not model_images:
        print(f"No images found for plotting {sampling_method}/{dataset}")
        return

    n_models = len(model_images)
    fig, axes = plt.subplots(n_models, 4, figsize=(12, 3 * n_models), constrained_layout=True)
    if n_models == 1:
        axes = axes[None, :]

    for idx, (model, files) in enumerate(model_images.items()):
        for col, (title, path) in enumerate(files.items()):
            ax = axes[idx, col]
            if path.suffix.lower() == '.npy':
                arr = np.load(path, allow_pickle=True)
                arr = np.squeeze(arr)
                ax.imshow(arr, cmap='gray')
            else:
                img = Image.open(path).convert("L")  # 흑백 변환 통일
                ax.imshow(img, cmap='gray')
            ax.axis('off')

            if idx == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')
            if col == 0:
                ax.text(-0.1, 0.5, model, transform=ax.transAxes,
                        rotation=90, va='center', fontsize=12, fontweight='bold')

    out_name = f"sample_{sampling_method}_{dataset}.jpg" if sampling_method else f"sample_{dataset}.jpg"
    plt.savefig(FIGURES_DIR / out_name, dpi=150)
    print(f"Plot saved: {FIGURES_DIR / out_name}")
    plt.close()


def plot_total_results(dataset: str, sampling_methods: List[str], models: List[str] = DEFAULT_MODELS, index: int = 0):
    """Generate comparison plots across sampling methods."""
    data_to_plot = {}
    valid_methods = set()

    for model in models:
        model_data = {}
        base_found = False

        for sm in sampling_methods:
            work_path = _get_workdir_path(dataset, model, sm)
            res_dirs = sorted(work_path.glob("results-*"))
            if not res_dirs:
                continue

            latest_dir = res_dirs[-1]

            images = natsorted(list(latest_dir.glob("*image*")))
            inputs = natsorted(list(latest_dir.glob("*input*")))
            targets = natsorted(list(latest_dir.glob("*target*")))
            outputs = natsorted(list(latest_dir.glob("*final_output*")))

            if not images or not inputs or not targets or not outputs:
                continue

            if index >= len(images) or index >= len(inputs) or index >= len(targets) or index >= len(outputs):
                continue

            # Get base images (Image, Target) from the first valid sampling method found
            if not base_found:
                img_path = images[index]
                input_path = inputs[index]
                tgt_path = targets[index]
                if img_path and input_path and tgt_path and img_path.exists() and input_path.exists() and tgt_path.exists():
                    model_data["Image"] = img_path
                    model_data["Input"] = input_path
                    model_data["Target"] = tgt_path
                    base_found = True

            # Get output for this sampling method
            out_path = outputs[index]
            if out_path and out_path.exists():
                model_data[sm] = out_path
                valid_methods.add(sm)

        if base_found and len(model_data) > 2:  # At least Image, Target and one Method
            data_to_plot[model] = model_data

    if not data_to_plot:
        print(f"No images found for total plotting {dataset}")
        return

    active_methods = [sm for sm in sampling_methods if sm in valid_methods]
    cols = ["Image", "Input", "Target"] + active_methods
    n_models = len(data_to_plot)
    n_cols = len(cols)

    fig, axes = plt.subplots(n_models, n_cols, figsize=(3 * n_cols, 3 * n_models), constrained_layout=True)
    fig.suptitle(f"Result for {dataset.replace('_', '-')}", fontsize=16, fontweight='bold')
    if n_models == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for idx, (model, files) in enumerate(data_to_plot.items()):
        for col_idx, col_name in enumerate(cols):
            ax = axes[idx, col_idx]

            if col_name in files:
                file_name = files[col_name]
                postfix = file_name.suffix.lower()
                if postfix == '.npy':
                    arr = np.load(file_name, allow_pickle=True)
                    arr = np.squeeze(arr)
                    ax.imshow(arr, cmap='gray')
                else:
                    img = Image.open(file_name).convert("L")
                    ax.imshow(img, cmap='gray')

            ax.axis('off')

            if idx == 0:
                title = col_name
                if col_name in active_methods:
                    title = col_name.replace('_', ' ').title()
                ax.set_title(title, fontsize=12, fontweight='bold')

            if col_idx == 0:
                ax.text(-0.1, 0.5, model, transform=ax.transAxes,
                        rotation=90, va='center', fontsize=12, fontweight='bold')

    out_name = f"total_sample_{dataset}.jpg"
    plt.savefig(FIGURES_DIR / out_name, dpi=150)
    print(f"Total Plot saved: {FIGURES_DIR / out_name}")
    plt.close()


def write_overall_latex_table(
    datasets: List[str],
    sampling_methods: List[str],
    models: List[str],
    prediction_type: str = "final",
):
    """Generate a comprehensive LaTeX table for all experiments."""
    hide_method = prediction_type == "input"
    lines = [
        "\\documentclass{article}",
        "\\usepackage{booktabs, amssymb, amsmath, graphicx}",
        "\\usepackage[a4paper,margin=1cm,landscape]{geometry}",
        "\\begin{document}",
    ]

    metric_goal = {m[0]: m[1] for m in METRICS}

    for dataset in datasets:
        lines.extend([
            "\\begin{table}[ht]",
            "\\centering",
            "{\\footnotesize",
            "\\begin{tabular}{" + ("l" if hide_method else "ll") + "c" * len(METRICS) + "}",
            "\\toprule",
            ("Model" if hide_method else "Method & Model") + " & " + " & ".join([m[0].replace('_', ' ').title() for m in METRICS]) + " " + "\\\\",
            "\\midrule"
        ])

        # Pre-calculate best and second best values for the dataset
        dataset_values = {m[0]: [] for m in METRICS}
        for sm in sampling_methods:
            for model in models:
                res = load_model_metrics(dataset, model, sm, prediction_type=prediction_type)
                if res:
                    for m_key, _ in METRICS:
                        val = res.get(m_key, 0)
                        mean, _ = _parse_metric_val(val)
                        dataset_values[m_key].append(mean)

        best_vals: Dict[str, Optional[float]] = {m[0]: None for m in METRICS}
        second_vals: Dict[str, Optional[float]] = {m[0]: None for m in METRICS}
        for m_key, vals in dataset_values.items():
            if not vals:
                continue
            maximize = metric_goal[m_key]
            unique_vals = sorted(list(set(vals)), reverse=maximize)
            best_vals[m_key] = unique_vals[0]
            second_vals[m_key] = unique_vals[1] if len(unique_vals) > 1 else None

        for sm in sampling_methods:
            valid_models = [m for m in models if load_model_metrics(dataset, m, sm, prediction_type=prediction_type)]
            if not valid_models:
                continue

            results_map = {}
            for model in valid_models:
                res = load_model_metrics(dataset, model, sm, prediction_type=prediction_type)
                results_map[model] = res

            for i, model in enumerate(valid_models):
                res = results_map[model]
                row = []

                if not hide_method:
                    if i == 0:
                        row.append(f"{sm}")
                    else:
                        row.append("")

                row.append(model)

                for m_key, _ in METRICS:
                    val = res.get(m_key, 0)
                    mean, std = _parse_metric_val(val)
                    val_str = f"{mean:.4f} \\pm {std:.4f}"

                    if best_vals[m_key] is not None and np.isclose(mean, best_vals[m_key]):
                        val_str = f"\\mathbf{{{val_str}}}"
                    elif second_vals[m_key] is not None and np.isclose(mean, second_vals[m_key]):
                        val_str = f"\\underline{{{val_str}}}"

                    row.append(f"${val_str}$")

                lines.append(" & ".join(row) + " \\\\")

            lines.append("\\midrule")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            f"\\caption{{Results for {dataset.replace('_', '-')}}}",
            "\\end{table}",
            "\\clearpage"
        ])

    lines.append("\\end{document}")

    suffix = "" if prediction_type == "final" else f"_{prediction_type}"
    tex_path = TEX_DIR / f"overall_results{suffix}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    _compile_latex(tex_path)


ALL_DATASETS = ["DRIVE", "OCTA500_3M", "OCTA500_6M"]
ALL_SAMPLING_METHODS = ["gaussian", "binomial", "poisson"]
ALL_PREDICTION_TYPES = list(PREDICTION_PATTERNS.keys())


@click.command()
@click.option("--dataset", required=True, type=click.Choice(["DRIVE", "OCTA500_3M", "OCTA500_6M", "all"]))
@click.option("--model", type=str, help="Specific model name.")
@click.option("--epoch", type=int, default=50000)
@click.option("--diffusion-type", type=str, default=None, help="Sampling method.")
@click.option("--evaluate-only", is_flag=True)
@click.option("--table-only", is_flag=True)
@click.option("--plot-only", is_flag=True)
@click.option(
    "--prediction-type",
    "prediction_types",
    type=click.Choice(list(PREDICTION_PATTERNS.keys())),
    multiple=True,
    help="Prediction source to evaluate (e.g., final, input). Use multiple times for several sources.",
)
@click.option("--delete-results", is_flag=True, help="Delete existing results before running.")
def main(dataset, model, epoch, diffusion_type, evaluate_only, table_only, plot_only, delete_results, prediction_types):
    """Main execution flow optimized for clarity."""

    plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.family'] = 'serif'

    # 설정 초기화
    datasets = ALL_DATASETS if dataset == "all" else [dataset]
    models = [model] if model else DEFAULT_MODELS
    sampling_methods = [diffusion_type] if diffusion_type else ALL_SAMPLING_METHODS
    prediction_types = list(dict.fromkeys(prediction_types)) if prediction_types else ALL_PREDICTION_TYPES

    # if delete_results 옵션 처리
    if delete_results and RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)

    # 필요한 디렉토리 생성
    for d in [RESULTS_DIR, WORKDIR_DIR, FIGURES_DIR, TEX_DIR, METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Itertools product를 사용하여 중첩 루프 평탄화
    configs = list(itertools.product(sampling_methods, datasets, models))
    input_configs = list(itertools.product(datasets, models))

    # 1. Evaluation Phase
    if not (table_only or plot_only):
        eval_tasks = [
            (sm, ds, md, "final")
            for sm, ds, md in configs
        ]
        t_eval = tqdm(eval_tasks, desc="Evaluating", unit="run", leave=False)
        set_desc_eval = getattr(t_eval, "set_description", None)
        for sm, ds, md, pred_type in t_eval:
            if set_desc_eval:
                set_desc_eval(f"Evaluating {sm}-{ds}-{md}-{pred_type}")
            evaluate_results(ds, md, epoch, sm, prediction_type=pred_type)

        input_tasks = [
            (ds, md, "input")
            for ds, md in input_configs
        ]
        t_input = tqdm(input_tasks, desc="Evaluating inputs", unit="run", leave=False)
        set_desc_input = getattr(t_input, "set_description", None)
        for ds, md, pred_type in t_input:
            if set_desc_input:
                set_desc_input(f"Evaluating {ds}-{md}-{pred_type}")
            evaluate_results(ds, md, epoch, prediction_type=pred_type)

    if evaluate_only:
        return

    # 2. Table & Plot Phase (Dataset & Sampling Method 단위로 그룹화 필요)
    # 중복 실행 방지를 위해 unique 조합 추출
    unique_ds_sm = sorted(set((c[1], c[0]) for c in configs))

    # Tables
    if not plot_only:
        table_tasks = [
            (ds, sm, pred_type)
            for (ds, sm) in unique_ds_sm
            for pred_type in prediction_types
        ]
        for ds, sm, pred_type in tqdm(table_tasks, desc="Tables", unit="job", leave=False):
            write_latex_table(ds, sm, models, prediction_type=pred_type)

    # Plots
    if not table_only:
        plot_tasks = list(unique_ds_sm)
        for ds, sm in tqdm(plot_tasks, desc="Plots", unit="cfg", leave=False):
            plot_results(ds, sm, models)

    if not plot_only and dataset == "all":
        for pred_type in tqdm(ALL_PREDICTION_TYPES, desc="Overall Tables", unit="type", leave=False):
            # Use sampling methods, not prediction types, for overall table aggregation
            write_overall_latex_table(ALL_DATASETS, ALL_SAMPLING_METHODS, DEFAULT_MODELS, prediction_type=pred_type)

    if not table_only and dataset == "all":
        for _ds in tqdm(ALL_DATASETS, desc="Overall Plots", unit="dataset", leave=False):
            plot_total_results(_ds, ALL_SAMPLING_METHODS, DEFAULT_MODELS)


if __name__ == "__main__":
    main()
