#!/usr/bin/env python3
"""Build dashboard figures with synchronized row-wise y-axis limits."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPTIBENCH_ROOT = DASHBOARD_ROOT.parent / "27-ICLR-OptiBench"
METRICS = (
    "quadratic_scale_distances",
    "angles",
    "relative_distances",
    "reference_norms",
    "validation_accuracies",
    "train_losses",
)
LANGUAGE_METRICS = (
    "quadratic_scale_distances",
    "angles",
    "relative_distances",
    "reference_norms",
    "validation_losses",
    "train_losses",
)
MOVING_AVERAGE_TYPES = ("simple", "exponential")
DEFAULT_MOVING_AVERAGE_TYPE = "exponential"
DEFAULT_EMA_LAMBDA = 0.95
NONNEGATIVE_METRICS = {
    "quadratic_scale_distances",
    "relative_distances",
    "reference_norms",
    "validation_losses",
    "train_losses",
}


@dataclass(frozen=True)
class FigureSpec:
    output: str
    title: str
    job_id: str
    result_dir: str
    labels: tuple[str, ...]
    csv_names: tuple[str, ...]
    excluded_csv_names: tuple[str, ...] = ()

    def csv_paths(self, optibench_root: Path) -> tuple[Path, ...]:
        root = (
            optibench_root
            / "results"
            / self.job_id
            / "source"
            / "results"
            / self.result_dir
        )
        return tuple(root / name for name in self.csv_names)


@dataclass(frozen=True)
class SectionSpec:
    name: str
    output_dir: str
    figures: tuple[FigureSpec, ...]
    metrics: tuple[str, ...] = METRICS
    window_size: int | None = None
    perplexity: bool = False


def optimizer_comparison(
    *,
    architecture: str,
    schedule: str,
    job_id: str,
    result_dir: str,
) -> FigureSpec:
    suffix = "fixed-lr" if schedule == "fixed" else "cosine-lr"
    display_architecture = {
        "resnet20": "ResNet-20",
        "resnet44": "ResNet-44",
    }[architecture]
    return FigureSpec(
        output=f"{architecture}-{suffix}",
        title=(
            f"CIFAR-10 {display_architecture} optimizer comparison: "
            f"SNE h=2 d=10 {'fixed' if schedule == 'fixed' else 'cosine'} LR"
        ),
        job_id=job_id,
        result_dir=result_dir,
        labels=("SGD", "AdamW", "Muon"),
        csv_names=(
            f"{architecture}_sgd_cifar10-gn4.csv",
            f"{architecture}_adamw_cifar10-gn4.csv",
            f"{architecture}_muon_cifar10-gn4.csv",
        ),
    )


def momentum_ablation(
    *,
    optimizer: str,
    schedule: str,
    job_id: str,
    result_dir: str,
    values: tuple[str, ...] = ("0.9", "0.5", "0.1"),
    excluded_csv_names: tuple[str, ...] = (),
) -> FigureSpec:
    suffix = "fixed-lr" if schedule == "fixed" else "cosine-lr"
    parameter = "beta1" if optimizer == "adamw" else "momentum"
    display = {"adamw": "AdamW", "muon": "Muon", "sgd": "SGD"}[optimizer]
    labels = tuple(f"{display} {parameter}={value}" for value in values)
    csv_names = tuple(
        f"resnet18_{optimizer}-{parameter}-{value.replace('.', '-')}_cifar10-gn4.csv"
        for value in values
    )
    return FigureSpec(
        output=f"{optimizer}-{suffix}",
        title=(
            f"CIFAR-10 ResNet-18 {display} {parameter} ablation: "
            f"SNE h=2 d=10 {'fixed' if schedule == 'fixed' else 'cosine'} LR"
        ),
        job_id=job_id,
        result_dir=result_dir,
        labels=labels,
        csv_names=csv_names,
        excluded_csv_names=excluded_csv_names,
    )


def shakespeare_comparison() -> FigureSpec:
    return FigureSpec(
        output="shakespeare-char-sne-m50",
        title="Character-level Shakespeare optimizer comparison: SNE h=10 d=5",
        job_id=(
            "nanogpt-shakespeare-char-optimizer-comparison-"
            "sne-m50-h10-d5-dropout02-2500-math-20260804-015300"
        ),
        result_dir=(
            "shakespeare-char/"
            "optimizer-comparison-sne-m50-h10-d5-dropout02-2500"
        ),
        labels=("SGD", "AdamW", "Muon"),
        csv_names=(
            "char-gpt_sgd_shakespeare-char.csv",
            "char-gpt_adamw_shakespeare-char.csv",
            "char-gpt_muon_shakespeare-char.csv",
        ),
    )


SECTIONS = (
    SectionSpec(
        name="optimizer-comparisons",
        output_dir="assets/plots/optimizer-comparisons",
        figures=(
            optimizer_comparison(
                architecture="resnet20",
                schedule="fixed",
                job_id=(
                    "cifar10-resnet20-optimizer-comparison-sne-h2-d10-"
                    "fixed-lr-20260803-150444"
                ),
                result_dir="cifar10-resnet20-sne-h2-d10-fixed-lr",
            ),
            optimizer_comparison(
                architecture="resnet44",
                schedule="fixed",
                job_id=(
                    "cifar10-resnet44-optimizer-comparison-sne-h2-d10-"
                    "fixed-lr-20260803-150600"
                ),
                result_dir="cifar10-resnet44-sne-h2-d10-fixed-lr",
            ),
            optimizer_comparison(
                architecture="resnet20",
                schedule="cosine",
                job_id=(
                    "cifar10-resnet20-optimizer-comparison-sne-h2-d10-"
                    "20260801-173942"
                ),
                result_dir="cifar10-resnet20-sne-h2-d10",
            ),
            optimizer_comparison(
                architecture="resnet44",
                schedule="cosine",
                job_id=(
                    "cifar10-resnet44-optimizer-comparison-sne-h2-d10-"
                    "20260801-174500"
                ),
                result_dir="cifar10-resnet44-sne-h2-d10",
            ),
        ),
    ),
    SectionSpec(
        name="momentum-ablations",
        output_dir="assets/plots/momentum-ablations",
        figures=(
            momentum_ablation(
                optimizer="muon",
                schedule="fixed",
                job_id=(
                    "cifar10-resnet18-muon-momentum-sne-h2-d10-fixed-lr-"
                    "20260803-151000"
                ),
                result_dir="cifar10-resnet18-muon-momentum-sne-h2-d10-fixed-lr",
            ),
            momentum_ablation(
                optimizer="adamw",
                schedule="fixed",
                job_id=(
                    "cifar10-resnet18-adamw-beta1-sne-h2-d10-fixed-lr-"
                    "20260803-151100"
                ),
                result_dir="cifar10-resnet18-adamw-beta1-sne-h2-d10-fixed-lr",
            ),
            momentum_ablation(
                optimizer="sgd",
                schedule="fixed",
                job_id=(
                    "cifar10-resnet18-sgd-momentum-sne-h2-d10-fixed-lr-"
                    "20260803-151200"
                ),
                result_dir="cifar10-resnet18-sgd-momentum-sne-h2-d10-fixed-lr",
            ),
            momentum_ablation(
                optimizer="muon",
                schedule="cosine",
                job_id=(
                    "cifar10-resnet18-muon-momentum-sne-h2-d10-cosine-lr-"
                    "20260804-014000"
                ),
                result_dir="cifar10-resnet18-muon-momentum-sne-h2-d10-cosine-lr",
            ),
            momentum_ablation(
                optimizer="adamw",
                schedule="cosine",
                job_id=(
                    "cifar10-resnet18-adamw-beta1-sne-h2-d10-cosine-lr-"
                    "20260804-014100"
                ),
                result_dir="cifar10-resnet18-adamw-beta1-sne-h2-d10-cosine-lr",
            ),
            momentum_ablation(
                optimizer="sgd",
                schedule="cosine",
                job_id=(
                    "cifar10-resnet18-sgd-momentum-sne-h2-d10-cosine-lr-"
                    "20260804-014200"
                ),
                result_dir="cifar10-resnet18-sgd-momentum-sne-h2-d10-cosine-lr",
            ),
        ),
    ),
    SectionSpec(
        name="language-modeling",
        output_dir="assets/plots/language-modeling",
        figures=(shakespeare_comparison(),),
        metrics=LANGUAGE_METRICS,
        window_size=100,
        perplexity=True,
    ),
)


def plotted_values(
    series: object,
    metric: str,
    window_size: int,
    *,
    perplexity: bool,
    moving_average_type: str,
    ema_lambda: float,
) -> tuple[float, ...]:
    from optibench.plot_training_results import moving_average

    raw_values = getattr(series, metric)
    if raw_values is None:
        raise ValueError(f"{series.optimizer} has no {metric} values")
    values = tuple(value for value in raw_values if math.isfinite(value))
    if not values:
        raise ValueError(f"{series.optimizer} has no finite {metric} values")
    if metric in {"validation_accuracies", "validation_losses"}:
        plotted = values
    else:
        plotted = moving_average(
            values,
            window_size,
            kind=moving_average_type,
            ema_lambda=ema_lambda,
        )
    if perplexity and metric in {"validation_losses", "train_losses"}:
        return tuple(math.exp(value) for value in plotted)
    return plotted


def padded_limits(metric: str, values: list[float]) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    span = upper - lower
    padding = 0.05 * span if span > 0.0 else max(1.0e-9, abs(lower) * 0.05)
    lower -= padding
    upper += padding
    if metric in NONNEGATIVE_METRICS:
        lower = max(0.0, lower)
    if metric == "angles":
        lower = max(0.0, lower)
        upper = min(math.pi, upper)
    if metric == "validation_accuracies":
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    if lower >= upper:
        upper = lower + max(1.0e-9, abs(lower) * 0.05)
    return lower, upper


def shared_limits(
    section: SectionSpec,
    optibench_root: Path,
    default_window_size: int,
    *,
    moving_average_type: str,
    ema_lambda: float,
) -> tuple[dict[str, tuple[float, float]], dict[str, list[object]]]:
    from optibench.plot_training_results import read_optimizer_series

    by_figure: dict[str, list[object]] = {}
    window_size = section.window_size or default_window_size
    values_by_metric = {metric: [] for metric in section.metrics}
    for figure in section.figures:
        paths = figure.csv_paths(optibench_root)
        missing = [path for path in paths if not path.is_file()]
        if missing:
            formatted = "\n".join(f"  {path}" for path in missing)
            raise FileNotFoundError(f"missing source CSVs:\n{formatted}")
        series = [
            read_optimizer_series(path, label)
            for label, path in zip(figure.labels, paths)
        ]
        by_figure[figure.output] = series
        for metric in section.metrics:
            for item in series:
                values_by_metric[metric].extend(
                    plotted_values(
                        item,
                        metric,
                        window_size,
                        perplexity=section.perplexity,
                        moving_average_type=moving_average_type,
                        ema_lambda=ema_lambda,
                    )
                )
    return (
        {
            metric: padded_limits(metric, values)
            for metric, values in values_by_metric.items()
        },
        by_figure,
    )


def build(
    optibench_root: Path,
    *,
    window_size: int = 100,
    moving_average_type: str = DEFAULT_MOVING_AVERAGE_TYPE,
    ema_lambda: float = DEFAULT_EMA_LAMBDA,
) -> None:
    sys.path.insert(0, str(optibench_root / "src"))
    from optibench.plot_training_results import plot_optimizer_comparison

    manifest: dict[str, object] = {
        "scale_error": "quadratic",
        "simple_moving_average_window_size": window_size,
        "moving_average_type": moving_average_type,
        "ema_lambda": ema_lambda,
        "section_simple_window_sizes": {
            section.name: section.window_size or window_size for section in SECTIONS
        },
        "validation_smoothed": False,
        "sections": {},
    }
    for section in SECTIONS:
        section_window_size = section.window_size or window_size
        limits, _ = shared_limits(
            section,
            optibench_root,
            window_size,
            moving_average_type=moving_average_type,
            ema_lambda=ema_lambda,
        )
        output_dir = DASHBOARD_ROOT / section.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        section_manifest = {
            "shared_y_limits": limits,
            "simple_moving_average_window_size": section_window_size,
            "moving_average_type": moving_average_type,
            "ema_lambda": ema_lambda,
            "perplexity": section.perplexity,
            "figures": [],
        }
        for figure in section.figures:
            paths = figure.csv_paths(optibench_root)
            output = plot_optimizer_comparison(
                list(figure.labels),
                list(paths),
                output_figure_name=figure.output,
                title=figure.title,
                output_dir=output_dir,
                window_size=section_window_size,
                moving_average_type=moving_average_type,
                ema_lambda=ema_lambda,
                perplexity=section.perplexity,
                scale_error="quadratic",
                y_limits=limits,
            )
            print(f"wrote {output.relative_to(DASHBOARD_ROOT)}")
            section_manifest["figures"].append(
                {
                    "output": output.relative_to(DASHBOARD_ROOT).as_posix(),
                    "job_id": figure.job_id,
                    "csvs": [
                        path.relative_to(optibench_root).as_posix() for path in paths
                    ],
                    "excluded_csvs": list(figure.excluded_csv_names),
                }
            )
        manifest["sections"][section.name] = section_manifest

    manifest_path = DASHBOARD_ROOT / "assets" / "plots" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path.relative_to(DASHBOARD_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optibench-root",
        type=Path,
        default=DEFAULT_OPTIBENCH_ROOT,
    )
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument(
        "--moving-average-type",
        choices=MOVING_AVERAGE_TYPES,
        default=DEFAULT_MOVING_AVERAGE_TYPE,
    )
    parser.add_argument(
        "--ema-lambda",
        type=float,
        default=DEFAULT_EMA_LAMBDA,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(
        arguments.optibench_root.expanduser().resolve(),
        window_size=arguments.window_size,
        moving_average_type=arguments.moving_average_type,
        ema_lambda=arguments.ema_lambda,
    )
