import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph

from dgraphfin import DGraphFin


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_split(mask, count):
    idx = mask.nonzero(as_tuple=False).view(-1)
    count = min(count, idx.numel())
    return idx[torch.randperm(idx.numel(), device=idx.device)[:count]]


def make_subgraph(data, train_count, val_count, test_count):
    train_nodes = sample_split(data.train_mask, train_count)
    val_nodes = sample_split(data.val_mask, val_count)
    test_nodes = sample_split(data.test_mask, test_count)
    selected = torch.cat([train_nodes, val_nodes, test_nodes]).cpu()

    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    node_mask[selected] = True
    edge_index_cpu = data.edge_index.cpu()
    incident_mask = node_mask[edge_index_cpu[0]] | node_mask[edge_index_cpu[1]]
    neighbor_nodes = torch.unique(edge_index_cpu[:, incident_mask])
    selected = torch.unique(torch.cat([selected, neighbor_nodes], dim=0))

    edge_index, _ = subgraph(selected, data.edge_index.cpu(), relabel_nodes=True, num_nodes=data.num_nodes)
    sub_data = data.__class__(x=data.x[selected].cpu(), y=data.y[selected].cpu(), edge_index=edge_index)

    local_index = {int(node_id): idx for idx, node_id in enumerate(selected.tolist())}
    train_local = torch.tensor([local_index[int(i)] for i in train_nodes.cpu().tolist()], dtype=torch.long)
    val_local = torch.tensor([local_index[int(i)] for i in val_nodes.cpu().tolist()], dtype=torch.long)
    test_local = torch.tensor([local_index[int(i)] for i in test_nodes.cpu().tolist()], dtype=torch.long)

    cursor = 0
    sub_data.train_mask = torch.zeros(selected.numel(), dtype=torch.bool)
    sub_data.train_mask[train_local] = True
    cursor += train_nodes.numel()
    sub_data.val_mask = torch.zeros(selected.numel(), dtype=torch.bool)
    sub_data.val_mask[val_local] = True
    cursor += val_nodes.numel()
    sub_data.test_mask = torch.zeros(selected.numel(), dtype=torch.bool)
    sub_data.test_mask[test_local] = True
    return sub_data


def attack_features(logits, y):
    probs = torch.softmax(logits, dim=1)
    top_probs = torch.topk(probs, k=2, dim=1).values
    max_conf = top_probs[:, 0]
    margin = top_probs[:, 0] - top_probs[:, 1]
    entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1) / np.log(probs.size(1))

    y_idx = y.long()
    row_idx = torch.arange(probs.size(0), device=probs.device)
    true_prob = probs[row_idx, y_idx].clamp(min=1e-12)
    true_logit = logits[row_idx, y_idx]
    loss = -torch.log(true_prob)
    return torch.cat([
        probs,
        logits,
        max_conf.unsqueeze(1),
        margin.unsqueeze(1),
        entropy.unsqueeze(1),
        true_prob.unsqueeze(1),
        true_logit.unsqueeze(1),
        loss.unsqueeze(1),
    ], dim=1)


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits[mask], dim=1)[:, 1].detach().cpu().numpy()
    pred = (probs >= 0.5).astype(int)
    y_true = data.y[mask].detach().cpu().numpy()
    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


@torch.no_grad()
def target_model_mia(model, data, max_samples, seed):
    model.eval()
    logits = model(data.x, data.edge_index)
    member_idx = data.train_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    nonmember_idx = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    rng = np.random.default_rng(seed)
    sample_count = min(max_samples, member_idx.shape[0], nonmember_idx.shape[0])
    member_idx = rng.choice(member_idx, size=sample_count, replace=False)
    nonmember_idx = rng.choice(nonmember_idx, size=sample_count, replace=False)

    idx = torch.from_numpy(np.concatenate([member_idx, nonmember_idx])).to(data.x.device)
    labels = np.concatenate([np.ones(sample_count), np.zeros(sample_count)])
    features = attack_features(logits[idx], data.y[idx]).detach().cpu().numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.5, stratify=labels, random_state=seed
    )
    attack = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    attack.fit(x_train, y_train)
    scores = attack.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, scores)), features.shape[1], float(scores[y_test == 1].mean()), float(scores[y_test == 0].mean())


def main():
    parser = argparse.ArgumentParser(description="Nonlinear GCN MIA stress test on DGraphFin")
    parser.add_argument("--data_dir", type=str, default=r"d:\experiment\PyG_datasets")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train_count", type=int, default=5000)
    parser.add_argument("--val_count", type=int, default=2000)
    parser.add_argument("--test_count", type=int, default=5000)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mia_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    dataset = DGraphFin(root=args.data_dir, name="DGraphFin")
    data = dataset[0]
    if hasattr(data, "valid_mask") and not hasattr(data, "val_mask"):
        data.val_mask = data.valid_mask
    if data.y.dim() > 1:
        data.y = data.y.squeeze(-1)
    data = make_subgraph(data, args.train_count, args.val_count, args.test_count)
    data.x = ((data.x - data.x.mean(dim=0)) / data.x.std(dim=0).clamp(min=1e-12)).float()
    data = data.to(device)

    print(f"Using device: {device}")
    print(f"Subgraph nodes={data.num_nodes}, edges={data.edge_index.size(1)}, train={int(data.train_mask.sum())}, val={int(data.val_mask.sum())}, test={int(data.test_mask.sum())}")
    print(f"Train labels={torch.bincount(data.y[data.train_mask], minlength=2).detach().cpu().tolist()}")
    print(f"Test labels={torch.bincount(data.y[data.test_mask], minlength=2).detach().cpu().tolist()}")

    model = GCN(data.x.size(1), args.hidden_channels, 2, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_counts = torch.bincount(data.y[data.train_mask], minlength=2).float().to(device)
    class_weight = class_counts.sum() / class_counts.clamp(min=1.0)
    class_weight = class_weight / class_weight.mean()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=class_weight)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_metrics = evaluate(model, data, data.train_mask)
            val_metrics = evaluate(model, data, data.val_mask)
            test_metrics = evaluate(model, data, data.test_mask)
            print(
                f"Epoch {epoch:03d} loss={loss.item():.4f} "
                f"train_auc={train_metrics['auc']:.4f} val_auc={val_metrics['auc']:.4f} test_auc={test_metrics['auc']:.4f} "
                f"test_f1={test_metrics['f1']:.4f} test_precision={test_metrics['precision']:.4f} test_recall={test_metrics['recall']:.4f}"
            )

    mia_auc, feature_dim, member_score, nonmember_score = target_model_mia(model, data, args.mia_samples, args.seed)
    print(f"[NONLINEAR MIA] feature_dim={feature_dim}, attack_auc={mia_auc:.4f}, member_score={member_score:.4f}, nonmember_score={nonmember_score:.4f}")


if __name__ == "__main__":
    main()
