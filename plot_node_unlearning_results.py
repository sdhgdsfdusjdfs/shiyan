from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import torch


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_1d_series(tensor: torch.Tensor, split_idx: int | None = None) -> torch.Tensor:
    t = tensor.detach().cpu()
    if t.ndim == 3:
        if split_idx is None:
            raise ValueError("split_idx is required for 3D tensors")
        t = t[split_idx]
    if t.ndim == 2:
        t = torch.nanmean(t, dim=1)
    return t.reshape(-1)


def plot_pair(x, y1, y2, title, ylabel, label1, label2, save_path: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.plot(x, y1, label=label1, linewidth=2)
    plt.plot(x, y2, label=label2, linewidth=2)
    plt.title(title)
    plt.xlabel("Removal Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot node unlearning metrics from a .pth result file.")
    parser.add_argument("--input", required=True, help="Path to the .pth result file")
    parser.add_argument("--output_dir", default="result/plots", help="Directory to save plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    result = torch.load(input_path, map_location="cpu")
    stem = input_path.stem

    x_auc = torch.arange(1, to_1d_series(result["auc_removal"], split_idx=1).numel() + 1)
    auc_removal = to_1d_series(result["auc_removal"], split_idx=1)
    auc_retrain = to_1d_series(result["auc_graph_retrain"], split_idx=1)
    plot_pair(
        x_auc,
        auc_removal,
        auc_retrain,
        "Node Unlearning Test AUC",
        "AUC",
        "Unlearning",
        "Retrain",
        output_dir / f"{stem}_test_auc.png",
    )

    x_f1 = torch.arange(1, to_1d_series(result["f1_removal"], split_idx=1).numel() + 1)
    f1_removal = to_1d_series(result["f1_removal"], split_idx=1)
    f1_retrain = to_1d_series(result["f1_graph_retrain"], split_idx=1)
    plot_pair(
        x_f1,
        f1_removal,
        f1_retrain,
        "Node Unlearning Test F1",
        "F1",
        "Unlearning",
        "Retrain",
        output_dir / f"{stem}_test_f1.png",
    )

    x_time = torch.arange(1, to_1d_series(result["removal_times"]).numel() + 1)
    removal_times = to_1d_series(result["removal_times"])
    retrain_times = to_1d_series(result["removal_times_graph_retrain"])
    plot_pair(
        x_time,
        removal_times,
        retrain_times,
        "Node Unlearning Runtime Per Step",
        "Seconds",
        "Unlearning",
        "Retrain",
        output_dir / f"{stem}_time.png",
    )

    print("Saved plots:")
    print(output_dir / f"{stem}_test_auc.png")
    print(output_dir / f"{stem}_test_f1.png")
    print(output_dir / f"{stem}_time.png")


if __name__ == "__main__":
    main()
