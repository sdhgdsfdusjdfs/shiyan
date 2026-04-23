from __future__ import annotations

import argparse
import random
from typing import List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import degree, subgraph

from dgraphfin import DGraphFin
from utils import MyGraphConv, preprocess_data


UndirectedEdge = Tuple[int, int]


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


def as_undirected(u: int, v: int) -> UndirectedEdge:
    return (u, v) if u < v else (v, u)


def collect_undirected_edges(edge_index: torch.Tensor) -> List[UndirectedEdge]:
    edges: Set[UndirectedEdge] = set()
    row = edge_index[0].detach().cpu().numpy()
    col = edge_index[1].detach().cpu().numpy()
    for u, v in zip(row, col):
        if u == v:
            continue
        edges.add(as_undirected(int(u), int(v)))
    return list(edges)


def remove_undirected_edges(edge_index: torch.Tensor, removed_edges: Set[UndirectedEdge]) -> torch.Tensor:
    row = edge_index[0].detach().cpu().numpy()
    col = edge_index[1].detach().cpu().numpy()
    keep_mask = []
    for u, v in zip(row, col):
        if u == v:
            keep_mask.append(True)
            continue
        keep_mask.append(as_undirected(int(u), int(v)) not in removed_edges)
    keep_mask_t = torch.tensor(keep_mask, dtype=torch.bool, device=edge_index.device)
    return edge_index[:, keep_mask_t]


def sample_removed_edges(
    edge_index: torch.Tensor,
    num_removes: int,
    seed: int,
    strategy: str = "random",
) -> List[UndirectedEdge]:
    rng = np.random.default_rng(seed)
    undirected_edges = collect_undirected_edges(edge_index)
    if not undirected_edges:
        return []
    k = min(num_removes, len(undirected_edges))
    if strategy == "random":
        idx = rng.choice(len(undirected_edges), size=k, replace=False)
        return [undirected_edges[i] for i in idx]
    if strategy == "high_degree":
        row = edge_index[0].detach().cpu()
        col = edge_index[1].detach().cpu()
        deg = degree(row, num_nodes=int(edge_index.max().item()) + 1) + degree(col, num_nodes=int(edge_index.max().item()) + 1)
        deg_np = deg.detach().cpu().numpy()
        scored = []
        for u, v in undirected_edges:
            score = float(deg_np[u] + deg_np[v])
            # jitter for tie-breaking while remaining reproducible
            score += float(rng.random() * 1e-8)
            scored.append((score, (u, v)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]
    raise ValueError(f"Unknown edge_delete_strategy: {strategy}")


def sample_non_edges(num_nodes: int, existing_edges: Set[UndirectedEdge], k: int, seed: int) -> List[UndirectedEdge]:
    rng = np.random.default_rng(seed)
    sampled: Set[UndirectedEdge] = set()
    max_trials = max(10000, 20 * k)
    trials = 0
    while len(sampled) < k and trials < max_trials:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        trials += 1
        if u == v:
            continue
        e = as_undirected(u, v)
        if e in existing_edges or e in sampled:
            continue
        sampled.add(e)
    return list(sampled)


def cosine_link_scores(X: torch.Tensor, edges: List[UndirectedEdge]) -> np.ndarray:
    if not edges:
        return np.array([], dtype=np.float64)
    idx_u = torch.tensor([u for u, _ in edges], dtype=torch.long, device=X.device)
    idx_v = torch.tensor([v for _, v in edges], dtype=torch.long, device=X.device)
    x_u = X[idx_u]
    x_v = X[idx_v]
    scores = F.cosine_similarity(x_u, x_v, dim=1).detach().cpu().numpy()
    return scores


def evaluate_link_inference(X: torch.Tensor, pos_edges: List[UndirectedEdge], neg_edges: List[UndirectedEdge]) -> Tuple[float, float]:
    pos_scores = cosine_link_scores(X, pos_edges)
    neg_scores = cosine_link_scores(X, neg_edges)
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    auc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score))
    return auc, ap


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight link inference audit for edge unlearning.")
    parser.add_argument("--data_dir", type=str, default=r"d:\experiment\PyG_datasets")
    parser.add_argument("--dataset", type=str, default="dgraphfin")
    parser.add_argument("--num_removes", type=int, default=500)
    parser.add_argument("--prop_step", type=int, default=2)
    parser.add_argument("--debug_sample_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--edge_delete_strategy",
        type=str,
        default="random",
        choices=["random", "high_degree"],
        help="Edge deletion strategy used in audit.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dataset.lower() != "dgraphfin":
        raise ValueError("This script currently supports only dgraphfin.")

    dataset = DGraphFin(root=args.data_dir, name="DGraphFin")
    data = dataset[0]
    if hasattr(data, "valid_mask") and not hasattr(data, "val_mask"):
        data.val_mask = data.valid_mask
    if data.y.dim() > 1:
        data.y = data.y.squeeze(-1)
    if args.debug_sample_size > 0:
        print(f"Applying debug subgraph sampling: {args.debug_sample_size}")
        data = maybe_sample_debug_subgraph(data, args.debug_sample_size)
    data = data.to(device)

    X0 = preprocess_data(data.x).float()
    propagation = MyGraphConv(K=args.prop_step, add_self_loops=True, alpha=0.0, XdegNorm=False, GPR=False).to(device)
    X_before = propagation(X0, data.edge_index) if args.prop_step > 0 else X0

    removed_edges = sample_removed_edges(
        data.edge_index,
        args.num_removes,
        args.seed,
        args.edge_delete_strategy,
    )
    removed_edge_set = set(removed_edges)
    edge_index_after = remove_undirected_edges(data.edge_index, removed_edge_set)
    X_after = propagation(X0, edge_index_after) if args.prop_step > 0 else X0

    existing_before = set(collect_undirected_edges(data.edge_index))
    neg_edges = sample_non_edges(data.num_nodes, existing_before, len(removed_edges), args.seed + 1)

    auc_before, ap_before = evaluate_link_inference(X_before, removed_edges, neg_edges)
    auc_after, ap_after = evaluate_link_inference(X_after, removed_edges, neg_edges)

    print(
        f"Nodes={data.num_nodes}, directed_edges={data.edge_index.size(1)}, "
        f"undirected_removed={len(removed_edges)}, strategy={args.edge_delete_strategy}"
    )
    print(f"[Link Inference Before] AUC={auc_before:.4f}, AP={ap_before:.4f}")
    print(f"[Link Inference After ] AUC={auc_after:.4f}, AP={ap_after:.4f}")
    print(f"Delta AUC (after-before)={auc_after - auc_before:.4f}, Delta AP={ap_after - ap_before:.4f}")


if __name__ == "__main__":
    main()
