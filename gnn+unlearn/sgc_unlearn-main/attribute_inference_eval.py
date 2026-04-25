from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import degree, subgraph

from dgraphfin import DGraphFin
from utils import MyGraphConv, preprocess_data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_sample_debug_subgraph(data: Data, debug_sample_size: int) -> Data:
    if debug_sample_size <= 0 or debug_sample_size >= data.num_nodes:
        return data

    split_names = ["train_mask", "val_mask", "test_mask"]
    split_masks = [getattr(data, name) for name in split_names]
    split_sizes = torch.tensor([int(mask.sum()) for mask in split_masks], dtype=torch.float)
    total_split = int(split_sizes.sum().item())
    if total_split == 0:
        return data

    target_total = min(debug_sample_size, total_split)
    raw_counts = torch.floor(split_sizes / total_split * target_total).to(torch.long)
    sampled_counts = [min(int(raw_counts[i].item()), int(split_sizes[i].item())) for i in range(len(split_masks))]

    remaining = target_total - sum(sampled_counts)
    split_order = torch.argsort(split_sizes, descending=True).tolist()
    while remaining > 0:
        updated = False
        for idx in split_order:
            if sampled_counts[idx] < int(split_sizes[idx].item()):
                sampled_counts[idx] += 1
                remaining -= 1
                updated = True
                if remaining == 0:
                    break
        if not updated:
            break

    undirected_degree = degree(data.edge_index[0], num_nodes=data.num_nodes) + degree(data.edge_index[1], num_nodes=data.num_nodes)

    selected_parts = []
    new_masks = {}
    cursor = 0
    for count, name, mask in zip(sampled_counts, split_names, split_masks):
        split_idx = mask.nonzero(as_tuple=False).view(-1)
        if count > 0:
            split_weights = undirected_degree[split_idx].float() + 1.0
            if count >= split_idx.size(0):
                picked = split_idx
            else:
                sampled_pos = torch.multinomial(split_weights, count, replacement=False)
                picked = split_idx[sampled_pos]
            selected_parts.append(picked)
            local_mask = torch.zeros(target_total, dtype=torch.bool, device=split_idx.device)
            local_mask[cursor:cursor + count] = True
            new_masks[name] = local_mask
            cursor += count
        else:
            new_masks[name] = torch.zeros(target_total, dtype=torch.bool, device=split_idx.device)

    selected_nodes = torch.cat(selected_parts, dim=0)
    sub_edge_index, edge_mask = subgraph(selected_nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

    sampled_data = Data(x=data.x[selected_nodes], edge_index=sub_edge_index, y=data.y[selected_nodes])
    for name, value in data.items():
        if name in ["x", "edge_index", "y"]:
            continue
        if isinstance(value, torch.Tensor):
            if value.size(0) == data.num_nodes:
                sampled_data[name] = value[selected_nodes]
            elif value.size(0) == data.edge_index.size(1):
                sampled_data[name] = value[edge_mask]
            else:
                sampled_data[name] = value
        else:
            sampled_data[name] = value

    for name in split_names:
        sampled_data[name] = new_masks[name]

    return sampled_data


def to_bool_mask(mask_like: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    if mask_like.dtype == torch.bool and mask_like.numel() == num_nodes:
        return mask_like.to(device)
    flat = mask_like.view(-1).to(device)
    if flat.dtype == torch.bool:
        if flat.numel() == num_nodes:
            return flat
        raise ValueError(f"Boolean mask length {flat.numel()} does not match num_nodes {num_nodes}.")
    out = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if flat.numel() > 0:
        if int(flat.min()) < 0 or int(flat.max()) >= num_nodes:
            raise ValueError("Index-style mask contains out-of-range node ids.")
        out[flat.long()] = True
    return out


def normalize_split_masks(data: Data) -> Data:
    num_nodes = int(data.num_nodes)
    device = data.x.device

    if hasattr(data, "train_mask"):
        data.train_mask = to_bool_mask(data.train_mask, num_nodes, device)
    elif hasattr(data, "train_idx"):
        data.train_mask = to_bool_mask(data.train_idx, num_nodes, device)
    else:
        raise ValueError("Missing train split in dataset.")

    if hasattr(data, "val_mask"):
        data.val_mask = to_bool_mask(data.val_mask, num_nodes, device)
    elif hasattr(data, "valid_mask"):
        data.val_mask = to_bool_mask(data.valid_mask, num_nodes, device)
    elif hasattr(data, "valid_idx"):
        data.val_mask = to_bool_mask(data.valid_idx, num_nodes, device)
    else:
        raise ValueError("Missing validation split in dataset.")

    if hasattr(data, "test_mask"):
        data.test_mask = to_bool_mask(data.test_mask, num_nodes, device)
    elif hasattr(data, "test_idx"):
        data.test_mask = to_bool_mask(data.test_idx, num_nodes, device)
    else:
        raise ValueError("Missing test split in dataset.")

    return data


def build_node_removal_queue(train_id: torch.Tensor, deg: torch.Tensor, strategy: str, seed: int) -> torch.Tensor:
    if strategy == "random":
        rng = torch.Generator(device=train_id.device)
        rng.manual_seed(seed)
        perm = torch.randperm(train_id.shape[0], generator=rng, device=train_id.device)
        return train_id[perm]
    if strategy == "high_degree":
        train_deg = deg[train_id].detach().cpu()
        ranked = torch.argsort(train_deg, descending=True)
        return train_id[ranked.to(train_id.device)]
    raise ValueError(f"Unsupported node_delete_strategy: {strategy}")


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _labels_for_dim(
    x_raw: torch.Tensor,
    dim: int,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> Tuple[torch.Tensor, float] | None:
    values = x_raw[:, dim]
    train_values = values[train_mask]
    threshold_candidates = [
        float(torch.median(train_values).item()),
        float(train_values.mean().item()),
        0.0,
    ]
    for th in threshold_candidates:
        labels = (values > th).long()
        train_classes = int(torch.unique(labels[train_mask]).numel())
        test_classes = int(torch.unique(labels[test_mask]).numel())
        if train_classes >= 2 and test_classes >= 2:
            return labels, th
    return None


def choose_sensitive_label(
    x_raw: torch.Tensor,
    preferred_dim: int,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> Tuple[torch.Tensor, int, float]:
    num_dims = int(x_raw.shape[1])
    checked = set()
    dim_order = []
    if 0 <= preferred_dim < num_dims:
        dim_order.append(preferred_dim)
        checked.add(preferred_dim)
    for dim in range(num_dims):
        if dim not in checked:
            dim_order.append(dim)

    for dim in dim_order:
        result = _labels_for_dim(x_raw, dim, train_mask, test_mask)
        if result is not None:
            labels, threshold = result
            return labels, dim, threshold

    raise RuntimeError("Cannot find a feature dimension with at least two classes in both train and removed sets.")


def evaluate_attack(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float, float, float, np.ndarray]:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_train, y_train)
    y_score = clf.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(np.int64)
    auc = safe_auc(y_test, y_score)
    ap = safe_ap(y_test, y_score)
    acc = float((y_pred == y_test).mean())
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    return auc, ap, acc, f1, y_score


def main() -> None:
    parser = argparse.ArgumentParser(description="AIA baseline for feature unlearning on DGraphFin.")
    parser.add_argument("--data_dir", type=str, default=r"d:\experiment\PyG_datasets")
    parser.add_argument("--dataset", type=str, default="dgraphfin")
    parser.add_argument("--num_removes", type=int, default=500)
    parser.add_argument("--node_delete_strategy", type=str, default="random", choices=["random", "high_degree"])
    parser.add_argument("--prop_step", type=int, default=2)
    parser.add_argument("--sensitive_dim", type=int, default=0, help="Feature column treated as sensitive attribute.")
    parser.add_argument(
        "--forget_mode",
        type=str,
        default="full_vector",
        choices=["full_vector", "sensitive_dim"],
        help="full_vector: zero all features of removed nodes; sensitive_dim: zero only sensitive feature.",
    )
    parser.add_argument("--debug_sample_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dataset.lower() != "dgraphfin":
        raise ValueError("This script currently supports only dgraphfin.")

    dataset = DGraphFin(root=args.data_dir, name="DGraphFin")
    data = dataset[0]
    if data.y.dim() > 1:
        data.y = data.y.squeeze(-1)
    data = normalize_split_masks(data)
    if args.debug_sample_size > 0:
        print(f"Applying debug subgraph sampling: {args.debug_sample_size}")
        data = maybe_sample_debug_subgraph(data, args.debug_sample_size)
    data = data.to(device)
    data = normalize_split_masks(data)

    if args.sensitive_dim < 0 or args.sensitive_dim >= int(data.x.shape[1]):
        raise ValueError(f"sensitive_dim must be in [0, {int(data.x.shape[1]) - 1}]")

    train_id = torch.arange(data.num_nodes, device=device)[data.train_mask]
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).to(device)
    removal_queue = build_node_removal_queue(train_id, deg, args.node_delete_strategy, args.seed)
    num_removes = min(args.num_removes, int(removal_queue.shape[0]))
    removed_nodes = removal_queue[:num_removes]
    removed_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    removed_mask[removed_nodes] = True

    x_raw = data.x.float()
    x_scaled = preprocess_data(x_raw).float()
    prop = MyGraphConv(K=args.prop_step, add_self_loops=True, alpha=0.0, XdegNorm=False, GPR=False).to(device)

    x_before = prop(x_scaled, data.edge_index) if args.prop_step > 0 else x_scaled
    x_scaled_after = x_scaled.clone().detach()
    if args.forget_mode == "full_vector":
        x_scaled_after[removed_nodes] = 0.0
    else:
        x_scaled_after[removed_nodes, args.sensitive_dim] = 0.0
    x_after = prop(x_scaled_after, data.edge_index) if args.prop_step > 0 else x_scaled_after

    attacker_train_mask = data.train_mask & (~removed_mask)
    attacker_test_mask = removed_mask
    if int(attacker_test_mask.sum()) == 0:
        raise RuntimeError("No removed nodes selected for attack test set.")
    sensitive_label, used_dim, threshold = choose_sensitive_label(
        x_raw, args.sensitive_dim, attacker_train_mask, attacker_test_mask
    )

    x_train_before = x_before[attacker_train_mask].detach().cpu().numpy()
    x_test_before = x_before[attacker_test_mask].detach().cpu().numpy()
    x_train_after = x_after[attacker_train_mask].detach().cpu().numpy()
    x_test_after = x_after[attacker_test_mask].detach().cpu().numpy()
    y_train = sensitive_label[attacker_train_mask].detach().cpu().numpy()
    y_test = sensitive_label[attacker_test_mask].detach().cpu().numpy()

    auc_before, ap_before, acc_before, f1_before, score_before = evaluate_attack(x_train_before, y_train, x_test_before, y_test)
    auc_after, ap_after, acc_after, f1_after, score_after = evaluate_attack(x_train_after, y_train, x_test_after, y_test)

    print(
        f"Nodes={data.num_nodes}, edges={data.edge_index.size(1)}, "
        f"removed={int(attacker_test_mask.sum())}, strategy={args.node_delete_strategy}, "
        f"sensitive_dim={used_dim}, threshold={threshold:.6f}, forget_mode={args.forget_mode}"
    )
    print(f"[AIA Before] AUC={auc_before:.4f}, AP={ap_before:.4f}, ACC={acc_before:.4f}, F1={f1_before:.4f}")
    print(f"[AIA After ] AUC={auc_after:.4f}, AP={ap_after:.4f}, ACC={acc_after:.4f}, F1={f1_after:.4f}")
    print(
        f"[AIA DIAG] score_std(before/after)=({float(np.std(score_before)):.6e}/{float(np.std(score_after)):.6e}), "
        f"unique_probs(before/after)=({int(np.unique(np.round(score_before, 12)).size)}/{int(np.unique(np.round(score_after, 12)).size)})"
    )
    if float(np.std(score_after)) < 1e-12:
        print("[AIA WARN] After-unlearning attack scores collapsed to a constant; AUC can be exactly 0.5.")
    print(
        "Delta(after-before): "
        f"AUC={auc_after - auc_before:.4f}, AP={ap_after - ap_before:.4f}, "
        f"ACC={acc_after - acc_before:.4f}, F1={f1_after - f1_before:.4f}"
    )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "seed,node_delete_strategy,num_removes,sensitive_dim,removed_nodes,"
            "auc_before,auc_after,ap_before,ap_after,acc_before,acc_after,f1_before,f1_after\n"
        )
        row = (
            f"{args.seed},{args.node_delete_strategy},{num_removes},{args.sensitive_dim},{int(attacker_test_mask.sum())},"
            f"{auc_before},{auc_after},{ap_before},{ap_after},{acc_before},{acc_after},{f1_before},{f1_after}\n"
        )
        if not out_path.exists():
            out_path.write_text(header + row, encoding="utf-8")
        else:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(row)
        print(f"Saved row to: {out_path}")


if __name__ == "__main__":
    main()
