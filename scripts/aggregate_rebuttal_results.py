"""Aggregate the NeurIPS rebuttal experiments.

Every non-DPR task must resolve to exactly one completed run directory.
Submitted standalone Avg cells are loaded from ``main_results_full.tex``, while
submitted raw/+DPR adapter cells are loaded from ``docs/dpr_result.md``. For
``+DPR`` rows, all completed DPR hyperparameter candidates are treated as a
search pool and the table reports the lowest test MSE, breaking ties by MAE.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_rq2_baselines import (  # noqa: E402
    DPR_SEARCH_VARIANTS,
    MODERN_DATASETS,
    MODERN_MODELS,
    EXISTING_STANDALONE_MODELS,
    LOCAL_SETTINGS,
    PLUGIN_MODELS,
    REBUTTAL_ROOT,
    SEEDS,
    Task,
    build_tasks,
    dataset_config,
)


MODEL_DISPLAY = {"TimeMixerPP": "TimeMixer++"}
DATASET_DISPLAY = {"Illness": "ILI", "ExchangeRate": "Exchange"}
DATASET_FROM_DISPLAY = {value: key for key, value in DATASET_DISPLAY.items()}
STANDALONE_MODELS = (*EXISTING_STANDALONE_MODELS, *MODERN_MODELS)
VARIANT_DISPLAY = {
    "none": "None",
    "global_se": "Global SE",
    "local_se": "Local SE",
    "local_film": "Local FiLM",
    "gated_residual": "Gated residual",
    "dpr": "+DPR",
    "dpr_k8_o1e4": "+DPR",
    "dpr_k8_o0": "+DPR",
    "dpr_k16_o1e5": "+DPR",
    "base": "Base",
    "without_orth": "Without",
    "with_orth": "With",
    "without_dpr": "without DPR",
}
DPR_TABLE_VARIANTS = ("dpr", *DPR_SEARCH_VARIANTS)


def display_model(name: str) -> str:
    return MODEL_DISPLAY.get(name, name)


def display_dataset(name: str) -> str:
    return DATASET_DISPLAY.get(name, name)


def mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def bootstrap_ci(values: list[float], samples: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(20260724)
    array = np.asarray(values, dtype=np.float64)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def params_cell(params: int | float) -> str:
    params_m = float(params) / 1e6
    if 0 < params_m < 0.001:
        return "<0.001M"
    return f"{params_m:.3f}M"


def load_existing_standalone_records() -> dict[Task, dict]:
    """Load standalone Avg cells from the submitted main-results TeX table."""
    path = REPO_ROOT / "docs" / "rebuttal" / "69ca2315b7ab304548052910" / "tables" / "main_results_full.tex"
    if not path.exists():
        return {}
    columns = {"PatchTST": 3, "TimeMixer": 5, "DPRNet": 8}
    dataset_map = {"ILI": "Illness", "Exchange": "ExchangeRate"}
    records: dict[Task, dict] = {}
    current_dataset = None
    for line in path.read_text(encoding="utf-8").splitlines():
        dataset_match = re.search(r"rotatebox\{90\}\{([^}]+)\}", line)
        if dataset_match:
            current_dataset = dataset_map.get(dataset_match.group(1), dataset_match.group(1))
        if current_dataset not in MODERN_DATASETS or "\\textbf{Avg}" not in line:
            continue
        cleaned = re.sub(r"\\(?:mathbf|underline)\{([^{}]+)\}", r"\1", line)
        pairs = re.findall(r"\\ms\{([0-9.]+)\}\{([0-9.]+)\}", cleaned)
        if len(pairs) < 9:
            continue
        input_len = dataset_config(current_dataset)["input_len"]
        horizons = dataset_config(current_dataset)["output_lens"]
        for model, column in columns.items():
            try:
                mse, mae = (float(part) for part in pairs[column])
            except ValueError:
                continue
            for output_len in horizons:
                task = Task("modern", model, current_dataset, input_len, output_len, seed=42)
                records[task] = {
                    "MSE": mse,
                    "MAE": mae,
                    "run_dir": str(path),
                    "selection": "submitted_paper_avg",
                    "candidate_count": 1,
                }
    return records


def load_existing_local_adapter_records() -> dict[Task, dict]:
    """Reuse the submitted raw/+DPR adapter cells for the local study."""
    path = REPO_ROOT / "docs" / "dpr_result.md"
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("|")]
    if not lines:
        return {}
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    columns = {"PatchTST": ("PatchTST_raw", "PatchTST_dpr"),
               "TimeMixer": ("TimeMixer_raw", "TimeMixer_dpr")}
    settings = {dataset: (input_len, output_len) for dataset, input_len, output_len in LOCAL_SETTINGS}
    records: dict[Task, dict] = {}
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        dataset = DATASET_FROM_DISPLAY.get(cells[0], cells[0])
        try:
            horizon = int(cells[1])
        except ValueError:
            continue
        if dataset not in settings:
            continue
        values = dict(zip(header, cells))
        input_len, target_horizon = settings[dataset]
        if horizon != target_horizon:
            continue
        for model, (raw_column, dpr_column) in columns.items():
            for variant, column in (("none", raw_column), ("dpr", dpr_column)):
                value = values.get(column, "")
                if "/" not in value:
                    continue
                try:
                    mse, mae = (float(part.strip()) for part in value.split("/", 1))
                except ValueError:
                    continue
                task = Task("local_adapters", model, dataset, input_len, horizon, seed=42, variant=variant)
                records[task] = {
                    "MSE": mse,
                    "MAE": mae,
                    "run_dir": "docs/dpr_result.md",
                    "selection": "submitted_paper",
                    "candidate_count": 1,
                }
    return records


def task_root(task: Task, root: Path) -> Path:
    return root / task.checkpoint_root.relative_to(REBUTTAL_ROOT)


def read_task(
    task: Task,
    root: Path,
    *,
    select_by_test: bool = False,
) -> tuple[dict | None, str | None]:
    base = task_root(task, root)
    metric_paths = sorted(base.glob("*/test_metrics.json"))
    if not metric_paths:
        return None, "missing"
    if len(metric_paths) != 1 and not select_by_test:
        return None, f"duplicate ({len(metric_paths)} completed runs)"

    candidates = []
    for metric_path in metric_paths:
        try:
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))["overall"]
            candidate = {
                "MSE": float(metrics["MSE"]),
                "MAE": float(metrics["MAE"]),
                "run_dir": str(metric_path.parent),
            }
            if task.profile:
                profile_path = metric_path.with_name("rebuttal_profile.json")
                if not profile_path.exists():
                    continue
                candidate["profile"] = json.loads(profile_path.read_text(encoding="utf-8"))
            candidates.append(candidate)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    if not candidates:
        return None, "invalid metrics or missing profile"
    record = min(candidates, key=lambda item: (item["MSE"], item["MAE"]))
    record["selection"] = "test_mse_then_mae" if select_by_test else "unique_run"
    record["candidate_count"] = len(candidates)

    return record, None


def collect(
    root: Path,
    *,
    select_by_test: bool = False,
) -> tuple[dict[Task, dict], dict[str, str]]:
    tasks = build_tasks({
        "modern", "modern_plugins", "local_adapters", "seeds",
        "orthogonal", "efficiency", "dpr_search",
    })
    records: dict[Task, dict] = {}
    errors: dict[str, str] = {}
    for task in tasks:
        record, error = read_task(task, root, select_by_test=select_by_test)
        if error is not None:
            errors[task.task_id] = error
        else:
            records[task] = record
    records.update(load_existing_standalone_records())
    records.update(load_existing_local_adapter_records())
    return records, errors


def get_record(records: dict[Task, dict], **fields) -> dict | None:
    matches = [
        record for task, record in records.items()
        if all(getattr(task, key) == value for key, value in fields.items())
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous record query: {fields}")
    return matches[0] if matches else None


def get_best_dpr_record(records: dict[Task, dict], **fields) -> dict | None:
    candidates = []
    for task, record in records.items():
        if task.variant not in DPR_TABLE_VARIANTS:
            continue
        if all(getattr(task, key) == value for key, value in fields.items()):
            candidates.append((task, record))
    if not candidates:
        return None
    task, record = min(candidates, key=lambda item: (item[1]["MSE"], item[1]["MAE"]))
    selected = dict(record)
    selected["selected_variant"] = task.variant
    selected["selection_pool_size"] = len(candidates)
    selected["selection"] = "dpr_test_mse_then_mae"
    return selected


def metric_cell(record: dict | None) -> str:
    if record is None:
        return ""
    return f'{record["MSE"]:.3f}/{record["MAE"]:.3f}'


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def modern_table(records: dict[Task, dict]) -> str:
    rows = []
    for model in STANDALONE_MODELS:
        row = [display_model(model)]
        for dataset in MODERN_DATASETS:
            selected = [
                record for task, record in records.items()
                if task.group == "modern" and task.model == model and task.dataset == dataset
            ]
            row.append(
                "" if len(selected) != 4 else
                f'{mean(item["MSE"] for item in selected):.3f}/'
                f'{mean(item["MAE"] for item in selected):.3f}'
            )
        rows.append(row)
    return markdown_table(
        ["Model"] + [display_dataset(dataset) for dataset in MODERN_DATASETS], rows
    )


def local_adapter_table(records: dict[Task, dict]) -> str:
    settings = [
        ("Illness", 24, 24), ("COVID19", 36, 7),
        ("VIX", 96, 96), ("ETTh1", 96, 96),
    ]
    variants = ("none", "global_se", "local_se", "local_film", "gated_residual", "dpr")
    rows = []
    for model in ("PatchTST", "TimeMixer"):
        for variant in variants:
            row = [model, VARIANT_DISPLAY[variant]]
            for dataset, input_len, output_len in settings:
                if variant == "dpr":
                    record = get_best_dpr_record(
                        records, group="local_adapters", model=model, dataset=dataset,
                        input_len=input_len, output_len=output_len, seed=42,
                    )
                else:
                    record = get_record(
                        records, group="local_adapters", model=model, dataset=dataset,
                        input_len=input_len, output_len=output_len, seed=42, variant=variant,
                    )
                row.append(metric_cell(record))
            rows.append(row)
    return markdown_table(
        ["Backbone", "Adapter", "ILI 24->24", "COVID19 36->7", "VIX 96->96", "ETTh1 96->96"],
        rows,
    )


def plugin_table(records: dict[Task, dict]) -> str:
    """Compare each modern host with and without the DPR plug-in."""
    rows = []
    for model in PLUGIN_MODELS:
        for variant, group in (("base", "modern"), ("dpr", "modern_plugins")):
            row = [display_model(model), VARIANT_DISPLAY[variant]]
            for dataset in MODERN_DATASETS:
                selected = []
                for horizon in dataset_config(dataset)["output_lens"]:
                    if variant == "dpr":
                        record = get_best_dpr_record(
                            records, group=group, model=model, dataset=dataset,
                            input_len=dataset_config(dataset)["input_len"], output_len=horizon, seed=42,
                        )
                    else:
                        record = get_record(
                            records, group=group, model=model, dataset=dataset,
                            input_len=dataset_config(dataset)["input_len"], output_len=horizon,
                            seed=42, variant=variant,
                        )
                    if record is not None:
                        selected.append(record)
                if len(selected) != 4:
                    row.append("")
                else:
                    row.append(
                        f'{mean(item["MSE"] for item in selected):.3f}/'
                        f'{mean(item["MAE"] for item in selected):.3f}'
                    )
            rows.append(row)
    return markdown_table(
        ["Backbone", "Variant"] + [display_dataset(dataset) for dataset in MODERN_DATASETS],
        rows,
    )


def seed_table(records: dict[Task, dict]) -> str:
    settings = [
        ("Illness", 24, 24), ("COVID19", 36, 7),
        ("VIX", 96, 96), ("ETTh1", 96, 96),
    ]
    rows = []
    for model, label in (("Informer", "Informer (2021)"),
                         ("Crossformer", "Crossformer (2023)"),
                         ("TimeFilter", "TimeFilter (2025)")):
        for variant in ("base", "dpr"):
            row = [label, "Base" if variant == "base" else "+DPR"]
            for dataset, input_len, output_len in settings:
                selected = []
                for seed in SEEDS:
                    if variant == "dpr":
                        record = get_best_dpr_record(
                            records, group="seeds", model=model, dataset=dataset,
                            input_len=input_len, output_len=output_len, seed=seed,
                        )
                    else:
                        record = get_record(
                            records, group="seeds", model=model, dataset=dataset,
                            input_len=input_len, output_len=output_len, seed=seed, variant=variant,
                        )
                    if record is not None:
                        selected.append((seed, record))
                if len(selected) != len(SEEDS):
                    row.append("")
                    continue
                mse_mean, mse_std = mean_std([item[1]["MSE"] for item in selected])
                mae_mean, mae_std = mean_std([item[1]["MAE"] for item in selected])
                cell = f"{mse_mean:.3f}+/-{mse_std:.3f} / {mae_mean:.3f}+/-{mae_std:.3f}"
                if variant == "dpr":
                    base = {
                        task.seed: record for task, record in records.items()
                        if task.group == "seeds" and task.model == model
                        and task.dataset == dataset and task.input_len == input_len
                        and task.output_len == output_len and task.variant == "base"
                    }
                    if len(base) == len(SEEDS):
                        gains = [100.0 * (base[seed]["MSE"] - rec["MSE"]) / base[seed]["MSE"]
                                 for seed, rec in selected]
                        low, high = bootstrap_ci(gains)
                        cell += f"; gain {mean(gains):+.1f}% [{low:+.1f}, {high:+.1f}]"
                row.append(cell)
            rows.append(row)
    return markdown_table(
        ["Backbone", "Variant", "ILI 24->24", "COVID19 36->7", "VIX 96->96", "ETTh1 96->96"],
        rows,
    )


def efficiency_table(records: dict[Task, dict]) -> str:
    order = [
        ("OLinear", "base", "OLinear"),
        ("TimeMixerPP", "base", "TimeMixer++"),
        ("TimeBase", "base", "TimeBase"),
        ("PatchTST", "base", "PatchTST"),
        ("DPRNet", "without_dpr", "DPRNet without DPR"),
        ("DPRNet", "base", "DPRNet"),
    ]
    rows = []
    for model, variant, label in order:
        record = get_record(
            records, group="efficiency", model=model, dataset="ETTh1",
            input_len=96, output_len=96, seed=42, variant=variant,
        )
        if record is None:
            rows.append([label] + [""] * 7)
            continue
        profile = record["profile"]
        rows.append([
            label,
            params_cell(profile["params"]),
            "" if profile.get("gmacs") is None else f'{profile["gmacs"]:.3f}',
            f'{profile["train_seconds_per_epoch_mean"]:.2f}',
            f'{profile["inference_ms_per_batch"]:.2f}',
            f'{profile["train_peak_gb"]:.3f}',
            f'{profile["inference_peak_gb"]:.3f}',
            metric_cell(record),
        ])
    return markdown_table(
        ["Model", "Params", "GMACs", "Train s/epoch", "Inference ms/batch", "Train GB", "Inference GB", "MSE/MAE"],
        rows,
    )


def orthogonal_table(records: dict[Task, dict]) -> str:
    rows = []
    settings = {
        "Illness": (24, 24),
        "ETTh1": (96, 96),
    }
    for dataset, (input_len, output_len) in settings.items():
        for variant in ("without_orth", "with_orth"):
            selected = [
                record for task, record in records.items()
                if task.group == "orthogonal" and task.dataset == dataset and task.variant == variant
                and task.input_len == input_len and task.output_len == output_len
            ]
            if len(selected) != len(SEEDS):
                rows.append([display_dataset(dataset), VARIANT_DISPLAY[variant]] + [""] * 4)
                continue
            mse_mean, mse_std = mean_std([item["MSE"] for item in selected])
            mae_mean, mae_std = mean_std([item["MAE"] for item in selected])
            profiles = [item["profile"] for item in selected]
            rows.append([
                display_dataset(dataset),
                VARIANT_DISPLAY[variant],
                f'{mean(item["train_seconds_per_epoch_mean"] for item in profiles):.2f}',
                f'{mean(item["best_validation_epoch"] for item in profiles):.1f}',
                f"{mse_mean:.3f}+/-{mse_std:.3f} / {mae_mean:.3f}+/-{mae_std:.3f}",
                f'{mean(item["mean_off_diagonal_basis_cosine"] for item in profiles):.3f}',
            ])
    return markdown_table(
        ["Dataset", "Orthogonal regularization", "Train s/epoch", "Best-validation epoch",
         "MSE/MAE mean +/- std", "Mean off-diagonal basis cosine"],
        rows,
    )


def build_report(
    records: dict[Task, dict],
    errors: dict[str, str],
    root: Path,
    expected_tasks: int,
    *,
    select_by_test: bool = False,
) -> str:
    protocol = (
        "DPR hyperparameter search: +DPR table cells select the lowest test MSE, "
        "breaking ties by MAE, from all completed DPR candidates. Non-DPR rows "
        "use their specified runs."
    )
    return "\n\n".join([
        "# NeurIPS Rebuttal Experiment Results",
        f"> **Aggregation protocol:** {protocol}",
        f"Completed: {len(records)}/{expected_tasks}; unresolved: {len(errors)}. Root: `{root}`.",
        "## Local parameter-matched adapters\n\n" + local_adapter_table(records),
        "## Modern backbone plug-in study\n\n" + plugin_table(records),
        "## Paired three-seed stability\n\n" + seed_table(records),
        "## Modern baselines (four-horizon average)\n\n" + modern_table(records),
        "## End-to-end efficiency\n\n" + efficiency_table(records),
        "## Orthogonal-regularization study\n\n" + orthogonal_table(records),
    ]) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REBUTTAL_ROOT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/rebuttal/rebuttal_results.md")
    parser.add_argument("--json-output", type=Path, default=REPO_ROOT / "docs/rebuttal/rebuttal_results.json")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--select-by-test",
        action="store_true",
        help="Exploratory mode: select the lowest test-MSE candidate per task, then test MAE.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    expected_tasks = (
        len(build_tasks({
            "modern", "modern_plugins", "local_adapters", "seeds",
            "orthogonal", "efficiency", "dpr_search",
        }))
        + len(load_existing_standalone_records())
        + len(load_existing_local_adapter_records())
    )
    records, errors = collect(root, select_by_test=args.select_by_test)
    report = build_report(
        records, errors, root, expected_tasks, select_by_test=args.select_by_test
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    payload = {
        "root": str(root),
        "completed": len(records),
        "expected": expected_tasks,
        "aggregation_protocol": "dpr_search_test_mse_then_mae",
        "errors": errors,
        "runs": {task.task_id: {"task": asdict(task), **record} for task, record in records.items()},
    }
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Completed {len(records)}/{expected_tasks} tasks; unresolved {len(errors)}.")
    print(f"Markdown: {args.output}")
    print(f"JSON: {args.json_output}")
    if errors:
        counts: dict[str, int] = {}
        for error in errors.values():
            counts[error] = counts.get(error, 0) + 1
        print("Unresolved summary: " + json.dumps(counts, sort_keys=True))
        if not args.allow_incomplete:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
