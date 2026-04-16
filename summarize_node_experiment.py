from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


FILE_RE = re.compile(
    r"dgraphfin_.*_nr_(?P<nr>\d+)_.*_bin_1(?:_(?P<strategy>random|high_degree))?_retrain\.pth$"
)


@dataclass
class RunSummary:
    nr: int
    strategy: str
    file_path: Path
    final_auc_unlearn: float
    final_auc_retrain: float
    final_f1_unlearn: float
    final_f1_retrain: float
    final_precision_unlearn: float
    final_precision_retrain: float
    final_recall_unlearn: float
    final_recall_retrain: float
    mean_step_time_unlearn: float
    mean_step_time_retrain: float
    speedup: float
    mia_before_mean: float
    mia_after_mean: float


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def build_summary(path: Path, nr: int, strategy: str) -> RunSummary:
    obj = torch.load(path, map_location="cpu", weights_only=False)

    auc_u = tensor_to_np(obj["auc_removal"][1, :, :])
    auc_r = tensor_to_np(obj["auc_graph_retrain"][1, :, :])
    f1_u = tensor_to_np(obj["f1_removal"][1, :, :])
    f1_r = tensor_to_np(obj["f1_graph_retrain"][1, :, :])
    p_u = tensor_to_np(obj["precision_removal"][1, :, :])
    p_r = tensor_to_np(obj["precision_graph_retrain"][1, :, :])
    r_u = tensor_to_np(obj["recall_removal"][1, :, :])
    r_r = tensor_to_np(obj["recall_graph_retrain"][1, :, :])
    t_u = tensor_to_np(obj["removal_times"])
    t_r = tensor_to_np(obj["removal_times_graph_retrain"])
    mia_b = tensor_to_np(obj.get("mia_auc_before_all", torch.full((1,), np.nan)))
    mia_a = tensor_to_np(obj.get("mia_auc_after_all", torch.full((1,), np.nan)))

    final_auc_u = float(np.nanmean(auc_u[-1, :]))
    final_auc_r = float(np.nanmean(auc_r[-1, :]))
    final_f1_u = float(np.nanmean(f1_u[-1, :]))
    final_f1_r = float(np.nanmean(f1_r[-1, :]))
    final_p_u = float(np.nanmean(p_u[-1, :]))
    final_p_r = float(np.nanmean(p_r[-1, :]))
    final_r_u = float(np.nanmean(r_u[-1, :]))
    final_r_r = float(np.nanmean(r_r[-1, :]))
    mean_t_u = float(np.nanmean(t_u))
    mean_t_r = float(np.nanmean(t_r))
    speedup = float(mean_t_r / max(mean_t_u, 1e-12))
    mia_b_mean = float(np.nanmean(mia_b))
    mia_a_mean = float(np.nanmean(mia_a))

    return RunSummary(
        nr=nr,
        strategy=strategy,
        file_path=path,
        final_auc_unlearn=final_auc_u,
        final_auc_retrain=final_auc_r,
        final_f1_unlearn=final_f1_u,
        final_f1_retrain=final_f1_r,
        final_precision_unlearn=final_p_u,
        final_precision_retrain=final_p_r,
        final_recall_unlearn=final_r_u,
        final_recall_retrain=final_r_r,
        mean_step_time_unlearn=mean_t_u,
        mean_step_time_retrain=mean_t_r,
        speedup=speedup,
        mia_before_mean=mia_b_mean,
        mia_after_mean=mia_a_mean,
    )


def discover_latest_files(root: Path, nrs: list[int]) -> dict[tuple[int, str], Path]:
    latest: dict[tuple[int, str], Path] = {}
    for p in root.rglob("dgraphfin*_bin_1*retrain.pth"):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        nr = int(m.group("nr"))
        if nr not in nrs:
            continue
        strategy = m.group("strategy") or "random"
        key = (nr, strategy)
        if key not in latest or p.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = p
    return latest


def write_csv(rows: list[RunSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "nr",
                "strategy",
                "file_path",
                "final_auc_unlearn",
                "final_auc_retrain",
                "final_f1_unlearn",
                "final_f1_retrain",
                "final_precision_unlearn",
                "final_precision_retrain",
                "final_recall_unlearn",
                "final_recall_retrain",
                "mean_step_time_unlearn",
                "mean_step_time_retrain",
                "speedup_retrain_over_unlearn",
                "mia_before_mean",
                "mia_after_mean",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.nr,
                    r.strategy,
                    str(r.file_path),
                    r.final_auc_unlearn,
                    r.final_auc_retrain,
                    r.final_f1_unlearn,
                    r.final_f1_retrain,
                    r.final_precision_unlearn,
                    r.final_precision_retrain,
                    r.final_recall_unlearn,
                    r.final_recall_retrain,
                    r.mean_step_time_unlearn,
                    r.mean_step_time_retrain,
                    r.speedup,
                    r.mia_before_mean,
                    r.mia_after_mean,
                ]
            )


def write_markdown(rows: list[RunSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Node Unlearning Summary",
        "",
        "| nr | strategy | AUC (U/R) | F1 (U/R) | Precision (U/R) | Recall (U/R) | Speedup | MIA (B/A) |",
        "|---:|:---------|:----------|:---------|:----------------|:-------------|--------:|:----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r.nr} | {r.strategy} | {r.final_auc_unlearn:.6f} / {r.final_auc_retrain:.6f} | "
            f"{r.final_f1_unlearn:.6f} / {r.final_f1_retrain:.6f} | "
            f"{r.final_precision_unlearn:.6f} / {r.final_precision_retrain:.6f} | "
            f"{r.final_recall_unlearn:.6f} / {r.final_recall_retrain:.6f} | "
            f"{r.speedup:.3f}x | {r.mia_before_mean:.4f} / {r.mia_after_mean:.4f} |"
        )
    lines.extend(["", "## Source Files", ""])
    for r in rows:
        lines.append(f"- nr={r.nr}, strategy={r.strategy}: `{r.file_path}`")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(rows: list[RunSummary], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda x: (x.nr, x.strategy))
    nrs = sorted({r.nr for r in rows})
    strategies = ["random", "high_degree"]
    x = np.arange(len(nrs))
    width = 0.36

    def values(metric: str, strategy: str) -> list[float]:
        mp = {(r.nr, r.strategy): getattr(r, metric) for r in rows}
        return [mp.get((nr, strategy), np.nan) for nr in nrs]

    overview_path = out_dir / "node_unlearning_overview.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    metrics = [
        ("final_auc_unlearn", "Test AUC (Unlearning)"),
        ("final_f1_unlearn", "Test F1 (Unlearning)"),
        ("speedup", "Speedup (Retrain/Unlearning)"),
        ("mia_after_mean", "MIA After"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for idx, strategy in enumerate(strategies):
            vals = values(metric, strategy)
            ax.bar(x + (idx - 0.5) * width, vals, width=width, label=strategy)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(nr) for nr in nrs])
        ax.set_xlabel("num_removes")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(overview_path, dpi=220)
    plt.close(fig)

    compare_path = out_dir / "node_unlearning_unlearn_vs_retrain.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for strategy in strategies:
        auc_u = values("final_auc_unlearn", strategy)
        auc_r = values("final_auc_retrain", strategy)
        f1_u = values("final_f1_unlearn", strategy)
        f1_r = values("final_f1_retrain", strategy)
        axes[0].plot(nrs, auc_u, marker="o", label=f"{strategy}-unlearn")
        axes[0].plot(nrs, auc_r, marker="x", linestyle="--", label=f"{strategy}-retrain")
        axes[1].plot(nrs, f1_u, marker="o", label=f"{strategy}-unlearn")
        axes[1].plot(nrs, f1_r, marker="x", linestyle="--", label=f"{strategy}-retrain")
    axes[0].set_title("AUC: Unlearning vs Retrain")
    axes[1].set_title("F1: Unlearning vs Retrain")
    for ax in axes:
        ax.set_xlabel("num_removes")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(compare_path, dpi=220)
    plt.close(fig)
    return overview_path, compare_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize node unlearning experiment results.")
    parser.add_argument("--root", default=r"d:\experiment", help="Workspace root to scan .pth files")
    parser.add_argument("--nrs", nargs="+", type=int, default=[100, 500, 1000], help="num_removes values to summarize")
    parser.add_argument("--out_dir", default=r"d:\experiment\result_exp1\summary", help="Output directory")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    latest = discover_latest_files(root, args.nrs)
    rows: list[RunSummary] = []
    for nr in sorted(args.nrs):
        for strategy in ["random", "high_degree"]:
            key = (nr, strategy)
            if key in latest:
                rows.append(build_summary(latest[key], nr, strategy))

    if not rows:
        raise SystemExit("No matching result files found.")

    rows = sorted(rows, key=lambda r: (r.nr, r.strategy))
    csv_path = out_dir / "node_unlearning_summary.csv"
    md_path = out_dir / "node_unlearning_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    overview_path, compare_path = make_plots(rows, out_dir)

    print("Generated:")
    print(csv_path)
    print(md_path)
    print(overview_path)
    print(compare_path)


if __name__ == "__main__":
    main()
