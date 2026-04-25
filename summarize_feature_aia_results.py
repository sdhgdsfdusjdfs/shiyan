from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _mean_std(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    g = df.groupby(["num_removes", "sensitive_dim", "node_delete_strategy"], as_index=False)[col].agg(["mean", "std"]).reset_index()
    return g["mean"].to_numpy(), g["std"].fillna(0.0).to_numpy()


def make_plot_dim500(summary: pd.DataFrame, out_path: Path) -> None:
    sub = summary[summary["num_removes"] == 500].copy()
    dims = sorted(sub["sensitive_dim"].unique().tolist())
    strategies = ["random", "high_degree"]
    x = np.arange(len(dims))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, title, ylabel in [
        (axes[0], "delta_auc_mean", "AIA Delta AUC by Sensitive Dim (nr=500)", "Delta AUC (after-before)"),
        (axes[1], "delta_ap_mean", "AIA Delta AP by Sensitive Dim (nr=500)", "Delta AP (after-before)"),
    ]:
        for i, st in enumerate(strategies):
            y = []
            e = []
            for d in dims:
                row = sub[(sub["sensitive_dim"] == d) & (sub["node_delete_strategy"] == st)]
                if row.empty:
                    y.append(np.nan)
                    e.append(0.0)
                else:
                    y.append(float(row[metric].iloc[0]))
                    e.append(float(row[metric.replace("_mean", "_std")].iloc[0]))
            ax.bar(x + (i - 0.5) * width, y, width, yerr=e, capsize=3, label=st)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("sensitive_dim")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_plot_nr_dim0(summary: pd.DataFrame, out_path: Path) -> None:
    sub = summary[summary["sensitive_dim"] == 0].copy()
    nrs = sorted(sub["num_removes"].unique().tolist())
    strategies = ["random", "high_degree"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, title, ylabel in [
        (axes[0], "delta_auc_mean", "AIA Delta AUC vs num_removes (dim=0)", "Delta AUC (after-before)"),
        (axes[1], "delta_ap_mean", "AIA Delta AP vs num_removes (dim=0)", "Delta AP (after-before)"),
    ]:
        for st in strategies:
            y = []
            e = []
            for nr in nrs:
                row = sub[(sub["num_removes"] == nr) & (sub["node_delete_strategy"] == st)]
                if row.empty:
                    y.append(np.nan)
                    e.append(0.0)
                else:
                    y.append(float(row[metric].iloc[0]))
                    e.append(float(row[metric.replace("_mean", "_std")].iloc[0]))
            ax.errorbar(nrs, y, yerr=e, marker="o", linewidth=2, capsize=3, label=st)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("num_removes")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and plot feature AIA results for paper-ready use.")
    parser.add_argument(
        "--in_csv",
        default=r"d:\experiment\shiyan\result_exp1\feature_aia_summary\aia_feature_runs_clean_for_paper.csv",
        help="Clean input csv",
    )
    parser.add_argument(
        "--out_dir",
        default=r"d:\experiment\shiyan\result_exp1\feature_aia_summary",
        help="Output directory",
    )
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    for m in ["auc", "ap", "acc", "f1"]:
        if f"delta_{m}" not in df.columns:
            df[f"delta_{m}"] = df[f"{m}_after"] - df[f"{m}_before"]

    summary = (
        df.groupby(["num_removes", "sensitive_dim", "node_delete_strategy"], as_index=False)
        .agg(
            n=("seed", "nunique"),
            auc_before_mean=("auc_before", "mean"),
            auc_before_std=("auc_before", "std"),
            auc_after_mean=("auc_after", "mean"),
            auc_after_std=("auc_after", "std"),
            delta_auc_mean=("delta_auc", "mean"),
            delta_auc_std=("delta_auc", "std"),
            ap_before_mean=("ap_before", "mean"),
            ap_before_std=("ap_before", "std"),
            ap_after_mean=("ap_after", "mean"),
            ap_after_std=("ap_after", "std"),
            delta_ap_mean=("delta_ap", "mean"),
            delta_ap_std=("delta_ap", "std"),
            acc_before_mean=("acc_before", "mean"),
            acc_before_std=("acc_before", "std"),
            acc_after_mean=("acc_after", "mean"),
            acc_after_std=("acc_after", "std"),
            delta_acc_mean=("delta_acc", "mean"),
            delta_acc_std=("delta_acc", "std"),
            f1_before_mean=("f1_before", "mean"),
            f1_before_std=("f1_before", "std"),
            f1_after_mean=("f1_after", "mean"),
            f1_after_std=("f1_after", "std"),
            delta_f1_mean=("delta_f1", "mean"),
            delta_f1_std=("delta_f1", "std"),
        )
        .fillna(0.0)
    )

    summary_csv = out_dir / "aia_feature_summary_with_plots.csv"
    summary_md = out_dir / "aia_feature_summary_with_plots.md"
    plot_dim500 = out_dir / "aia_feature_delta_by_sensitive_dim_nr500.png"
    plot_nr_dim0 = out_dir / "aia_feature_delta_by_num_removes_dim0.png"

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    lines = [
        "# Feature AIA Summary (Plot Version)",
        "",
        "| num_removes | sensitive_dim | strategy | n | delta_auc(mean±std) | delta_ap(mean±std) |",
        "|---:|---:|:---|---:|---:|---:|",
    ]
    for _, r in summary.sort_values(["num_removes", "sensitive_dim", "node_delete_strategy"]).iterrows():
        lines.append(
            f"| {int(r['num_removes'])} | {int(r['sensitive_dim'])} | {r['node_delete_strategy']} | {int(r['n'])} | "
            f"{r['delta_auc_mean']:.4f} ± {r['delta_auc_std']:.4f} | "
            f"{r['delta_ap_mean']:.4f} ± {r['delta_ap_std']:.4f} |"
        )
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    make_plot_dim500(summary, plot_dim500)
    make_plot_nr_dim0(summary, plot_nr_dim0)

    print("Generated:")
    print(summary_csv)
    print(summary_md)
    print(plot_dim500)
    print(plot_nr_dim0)


if __name__ == "__main__":
    main()
