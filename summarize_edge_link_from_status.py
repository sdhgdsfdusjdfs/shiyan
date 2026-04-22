from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLOCK_RE = re.compile(
    r"Nodes=(?P<nodes>\d+),\s*directed_edges=(?P<directed>\d+),\s*undirected_removed=(?P<num_removes>\d+)\s*"
    r"\[Link Inference Before\]\s*AUC=(?P<before_auc>[0-9.]+),\s*AP=(?P<before_ap>[0-9.]+)\s*"
    r"\[Link Inference After \]\s*AUC=(?P<after_auc>[0-9.]+),\s*AP=(?P<after_ap>[0-9.]+)\s*"
    r"Delta AUC \(after-before\)=(?P<delta_auc>-?[0-9.]+),\s*Delta AP=(?P<delta_ap>-?[0-9.]+)",
    re.S,
)


@dataclass
class Row:
    num_removes: int
    run_index: int
    before_auc: float
    after_auc: float
    delta_auc: float
    before_ap: float
    after_ap: float
    delta_ap: float


def parse_rows(status_file: Path) -> list[Row]:
    text = status_file.read_text(encoding="utf-8", errors="ignore")
    matches = list(BLOCK_RE.finditer(text))
    run_counter: dict[int, int] = defaultdict(int)
    rows: list[Row] = []
    for m in matches:
        nr = int(m.group("num_removes"))
        run_counter[nr] += 1
        rows.append(
            Row(
                num_removes=nr,
                run_index=run_counter[nr],
                before_auc=float(m.group("before_auc")),
                after_auc=float(m.group("after_auc")),
                delta_auc=float(m.group("delta_auc")),
                before_ap=float(m.group("before_ap")),
                after_ap=float(m.group("after_ap")),
                delta_ap=float(m.group("delta_ap")),
            )
        )
    return rows


def write_csv(rows: list[Row], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "num_removes",
                "run_index",
                "before_auc",
                "after_auc",
                "delta_auc",
                "before_ap",
                "after_ap",
                "delta_ap",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.num_removes,
                    r.run_index,
                    r.before_auc,
                    r.after_auc,
                    r.delta_auc,
                    r.before_ap,
                    r.after_ap,
                    r.delta_ap,
                ]
            )


def write_markdown(rows: list[Row], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Edge Link-Inference Summary (From Status File)",
        "",
        "| num_removes | run | AUC Before | AUC After | Delta AUC | AP Before | AP After | Delta AP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.num_removes} | {r.run_index} | {r.before_auc:.4f} | {r.after_auc:.4f} | {r.delta_auc:.4f} | "
            f"{r.before_ap:.4f} | {r.after_ap:.4f} | {r.delta_ap:.4f} |"
        )

    lines.extend(["", "## Mean +/- Std by num_removes", ""])
    groups: dict[int, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.num_removes].append(r)

    lines.append("| num_removes | Delta AUC mean+/-std | Delta AP mean+/-std |")
    lines.append("|---:|---:|---:|")
    for nr in sorted(groups):
        da = np.array([x.delta_auc for x in groups[nr]], dtype=float)
        dp = np.array([x.delta_ap for x in groups[nr]], dtype=float)
        lines.append(
            f"| {nr} | {da.mean():.4f} +/- {da.std(ddof=1) if len(da) > 1 else 0.0:.4f} | "
            f"{dp.mean():.4f} +/- {dp.std(ddof=1) if len(dp) > 1 else 0.0:.4f} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def make_plots(rows: list[Row], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[int, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.num_removes].append(r)

    nrs = sorted(groups.keys())
    x = np.arange(len(nrs))
    width = 0.36

    auc_before = [np.mean([r.before_auc for r in groups[nr]]) for nr in nrs]
    auc_after = [np.mean([r.after_auc for r in groups[nr]]) for nr in nrs]
    ap_before = [np.mean([r.before_ap for r in groups[nr]]) for nr in nrs]
    ap_after = [np.mean([r.after_ap for r in groups[nr]]) for nr in nrs]

    auc_before_std = [np.std([r.before_auc for r in groups[nr]], ddof=1) if len(groups[nr]) > 1 else 0.0 for nr in nrs]
    auc_after_std = [np.std([r.after_auc for r in groups[nr]], ddof=1) if len(groups[nr]) > 1 else 0.0 for nr in nrs]
    ap_before_std = [np.std([r.before_ap for r in groups[nr]], ddof=1) if len(groups[nr]) > 1 else 0.0 for nr in nrs]
    ap_after_std = [np.std([r.after_ap for r in groups[nr]], ddof=1) if len(groups[nr]) > 1 else 0.0 for nr in nrs]

    metric_path = out_dir / "edge_link_inference_before_after.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(x - width / 2, auc_before, width, yerr=auc_before_std, capsize=4, label="Before")
    axes[0].bar(x + width / 2, auc_after, width, yerr=auc_after_std, capsize=4, label="After")
    axes[0].set_title("Link Inference AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(nr) for nr in nrs])
    axes[0].set_xlabel("num_removes")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - width / 2, ap_before, width, yerr=ap_before_std, capsize=4, label="Before")
    axes[1].bar(x + width / 2, ap_after, width, yerr=ap_after_std, capsize=4, label="After")
    axes[1].set_title("Link Inference AP")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(nr) for nr in nrs])
    axes[1].set_xlabel("num_removes")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(metric_path, dpi=220)
    plt.close(fig)

    delta_auc_mean = [np.mean([r.delta_auc for r in groups[nr]]) for nr in nrs]
    delta_ap_mean = [np.mean([r.delta_ap for r in groups[nr]]) for nr in nrs]
    delta_auc_std = [np.std([r.delta_auc for r in groups[nr]], ddof=1) if len(groups[nr]) > 1 else 0.0 for nr in nrs]
    delta_ap_std = [np.std([r.delta_ap for r in groups[nr]], ddof=1) if len(groups[nr]) > 1 else 0.0 for nr in nrs]

    delta_path = out_dir / "edge_link_inference_delta.png"
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.6))
    ax.errorbar(nrs, delta_auc_mean, yerr=delta_auc_std, marker="o", linewidth=2, capsize=4, label="Delta AUC")
    ax.errorbar(nrs, delta_ap_mean, yerr=delta_ap_std, marker="s", linewidth=2, capsize=4, label="Delta AP")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Privacy Gain (After - Before)")
    ax.set_xlabel("num_removes")
    ax.set_ylabel("Delta")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(delta_path, dpi=220)
    plt.close(fig)
    return metric_path, delta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract edge link-inference results from status file and draw plots.")
    parser.add_argument("--status_file", default=r"d:\experiment\shiyan\下一步.txt", help="Path to status text file")
    parser.add_argument("--out_dir", default=r"d:\experiment\shiyan\result_exp1\edge_link_summary", help="Output directory")
    args = parser.parse_args()

    status_file = Path(args.status_file)
    out_dir = Path(args.out_dir)
    rows = parse_rows(status_file)
    if not rows:
        raise SystemExit(f"No link-inference result blocks found in {status_file}")

    rows.sort(key=lambda r: (r.num_removes, r.run_index))
    csv_path = out_dir / "edge_link_inference_runs.csv"
    md_path = out_dir / "edge_link_inference_summary.md"
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
