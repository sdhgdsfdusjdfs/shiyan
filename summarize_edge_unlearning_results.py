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
    r"dgraphfin_.*_nr_(?P<nr>\d+)_.*_mode_edge_.*_bin_1(?:_estrat_(?P<strategy>random|high_degree))?_retrain\.pth$"
)


@dataclass
class EdgeSummary:
    nr: int
    strategy: str
    file_path: Path
    final_auc_unlearn: float
    final_auc_retrain: float
    final_f1_unlearn: float
    final_f1_retrain: float
    mean_time_unlearn: float
    mean_time_retrain: float
    speedup: float


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def build_summary(path: Path) -> EdgeSummary:
    m = FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized edge result filename: {path.name}")
    nr = int(m.group("nr"))
    strategy = m.group("strategy") or "random"

    obj = torch.load(path, map_location="cpu", weights_only=False)
    auc_u = to_np(obj["auc_removal"][1, :, :])
    auc_r = to_np(obj["auc_graph_retrain"][1, :, :])
    f1_u = to_np(obj["f1_removal"][1, :, :])
    f1_r = to_np(obj["f1_graph_retrain"][1, :, :])
    t_u = to_np(obj["removal_times"])
    t_r = to_np(obj["removal_times_graph_retrain"])

    final_auc_u = float(np.nanmean(auc_u[-1, :]))
    final_auc_r = float(np.nanmean(auc_r[-1, :]))
    final_f1_u = float(np.nanmean(f1_u[-1, :]))
    final_f1_r = float(np.nanmean(f1_r[-1, :]))
    mean_t_u = float(np.nanmean(t_u))
    mean_t_r = float(np.nanmean(t_r))
    speedup = float(mean_t_r / max(mean_t_u, 1e-12))

    return EdgeSummary(
        nr=nr,
        strategy=strategy,
        file_path=path,
        final_auc_unlearn=final_auc_u,
        final_auc_retrain=final_auc_r,
        final_f1_unlearn=final_f1_u,
        final_f1_retrain=final_f1_r,
        mean_time_unlearn=mean_t_u,
        mean_time_retrain=mean_t_r,
        speedup=speedup,
    )


def write_csv(rows: list[EdgeSummary], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "num_removes",
                "strategy",
                "file_path",
                "final_auc_unlearn",
                "final_auc_retrain",
                "final_f1_unlearn",
                "final_f1_retrain",
                "mean_time_unlearn",
                "mean_time_retrain",
                "speedup_retrain_over_unlearn",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.nr,
                    r.strategy,
                    str(r.file_path),
                    r.final_auc_unlearn,
                    r.final_auc_retrain,
                    r.final_f1_unlearn,
                    r.final_f1_retrain,
                    r.mean_time_unlearn,
                    r.mean_time_retrain,
                    r.speedup,
                ]
            )


def write_markdown(rows: list[EdgeSummary], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Edge Unlearning Utility/Efficiency Summary",
        "",
        "| num_removes | strategy | AUC (U/R) | F1 (U/R) | mean time (U/R, s) | speedup |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.nr} | {r.strategy} | {r.final_auc_unlearn:.4f} / {r.final_auc_retrain:.4f} | "
            f"{r.final_f1_unlearn:.4f} / {r.final_f1_retrain:.4f} | "
            f"{r.mean_time_unlearn:.4f} / {r.mean_time_retrain:.4f} | "
            f"{r.speedup:.3f}x |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def make_plots(rows: list[EdgeSummary], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r.nr)
    nrs = [r.nr for r in rows]

    # Plot 1: utility (AUC/F1)
    utility_path = out_dir / "edge_unlearning_utility_unlearn_vs_retrain.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(nrs, [r.final_auc_unlearn for r in rows], marker="o", linewidth=2, label="Unlearning")
    axes[0].plot(nrs, [r.final_auc_retrain for r in rows], marker="x", linestyle="--", linewidth=2, label="Retrain")
    axes[0].set_title("Edge Unlearning Test AUC")
    axes[0].set_xlabel("num_removes")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(nrs, [r.final_f1_unlearn for r in rows], marker="o", linewidth=2, label="Unlearning")
    axes[1].plot(nrs, [r.final_f1_retrain for r in rows], marker="x", linestyle="--", linewidth=2, label="Retrain")
    axes[1].set_title("Edge Unlearning Test F1")
    axes[1].set_xlabel("num_removes")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(utility_path, dpi=220)
    plt.close(fig)

    # Plot 2: efficiency (time + speedup)
    efficiency_path = out_dir / "edge_unlearning_efficiency_time_speedup.png"
    x = np.arange(len(nrs))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
    ax1.bar(x - width / 2, [r.mean_time_unlearn for r in rows], width, label="Unlearning time")
    ax1.bar(x + width / 2, [r.mean_time_retrain for r in rows], width, label="Retrain time")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(nr) for nr in nrs])
    ax1.set_xlabel("num_removes")
    ax1.set_ylabel("mean step time (s)")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, [r.speedup for r in rows], color="black", marker="o", linewidth=2, label="Speedup")
    ax2.set_ylabel("speedup (retrain/unlearning)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    plt.title("Edge Unlearning Efficiency")
    fig.tight_layout()
    fig.savefig(efficiency_path, dpi=220)
    plt.close(fig)

    return utility_path, efficiency_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize edge-unlearning utility/efficiency and draw 2 plots.")
    parser.add_argument(
        "--result_dir",
        default=r"d:\experiment\shiyan\gnn+unlearn\sgc_unlearn-main\result_edge",
        help="Directory containing edge result .pth files",
    )
    parser.add_argument(
        "--nrs",
        nargs="+",
        type=int,
        default=[500, 1000],
        help="num_removes values to include",
    )
    parser.add_argument(
        "--out_dir",
        default=r"d:\experiment\shiyan\result_exp1\edge_link_summary",
        help="Output directory for summary and plots",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "high_degree", "all"],
        help="Filter files by deletion strategy; use all to include both.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)

    rows: list[EdgeSummary] = []
    for p in sorted(result_dir.glob("dgraphfin*_mode_edge*.pth")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        row = build_summary(p)
        if row.nr in args.nrs and (args.strategy == "all" or row.strategy == args.strategy):
            rows.append(row)
    rows.sort(key=lambda r: (r.nr, r.strategy))

    if not rows:
        raise SystemExit(f"No matching edge result files found in {result_dir}")

    csv_path = out_dir / "edge_unlearning_utility_efficiency.csv"
    md_path = out_dir / "edge_unlearning_utility_efficiency.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    plot1, plot2 = make_plots(rows, out_dir)

    print("Generated:")
    print(csv_path)
    print(md_path)
    print(plot1)
    print(plot2)


if __name__ == "__main__":
    main()
