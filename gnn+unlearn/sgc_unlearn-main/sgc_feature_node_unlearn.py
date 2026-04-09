from __future__ import print_function
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# Below is for graph learning part
from torch_geometric.nn.conv import MessagePassing
from typing import Optional

from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree, subgraph

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp

from torch.nn import init
from utils import *
from dgraphfin import DGraphFin  # ← 使用自定义加载器替代官方版本

from sklearn import preprocessing
from numpy.linalg import norm

from utils import membership_inference_attack


def predict_from_weights(w, X):
    with torch.no_grad():
        if w.dim() == 1:
            logits = X.mv(w)
            probs_pos = torch.sigmoid(logits)
            preds = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))
            scores = probs_pos.unsqueeze(1)
        else:
            logits = X.mm(w)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            scores = probs
    return preds, scores


def evaluate_metrics(w, X, y, train_mode):
    preds, scores = predict_from_weights(w, X)

    if train_mode == 'binary':
        y_true = y.detach().cpu().numpy()
        y_true_binary = (y_true == 1).astype(int)
        pred_binary = (preds.detach().cpu().numpy() == 1).astype(int)
        score_np = scores.detach().cpu().numpy().reshape(-1)
        acc = float((preds == y).float().mean().item())
        f1 = float(f1_score(y_true_binary, pred_binary, average='binary'))
        try:
            auc = float(roc_auc_score(y_true_binary, score_np))
        except ValueError:
            auc = float('nan')
    else:
        y_true = y.detach().cpu().numpy()
        pred_np = preds.detach().cpu().numpy()
        score_np = scores.detach().cpu().numpy()
        acc = float((preds == y).float().mean().item())
        f1 = float(f1_score(y_true, pred_np, average='macro'))
        try:
            if score_np.shape[1] == 2:
                auc = float(roc_auc_score(y_true, score_np[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, score_np, multi_class='ovr', average='macro'))
        except ValueError:
            auc = float('nan')

    return acc, f1, auc


def print_metric_summary(prefix, metrics):
    acc, f1, auc = metrics
    print(f"{prefix} accuracy = {acc:.4f}, F1 = {f1:.4f}, AUC = {auc:.4f}")


def print_label_distribution(prefix, y, num_classes):
    y_cpu = y.detach().cpu().long()
    counts = torch.bincount(y_cpu, minlength=num_classes)
    print(f"{prefix} label distribution: {counts.tolist()}")


def print_prediction_distribution(prefix, w, X, num_classes):
    preds, _ = predict_from_weights(w, X)
    pred_cpu = preds.detach().cpu().long()
    counts = torch.bincount(pred_cpu, minlength=num_classes)
    print(f"{prefix} prediction distribution: {counts.tolist()}")


def print_binary_classification_details(prefix, w, X, y):
    preds, scores = predict_from_weights(w, X)
    y_true = (y.detach().cpu().numpy() == 1).astype(int)
    pred_binary = (preds.detach().cpu().numpy() == 1).astype(int)
    pos_rate = float(scores.detach().cpu().numpy().reshape(-1).mean())
    precision = float(precision_score(y_true, pred_binary, zero_division=0))
    recall = float(recall_score(y_true, pred_binary, zero_division=0))
    print(f"{prefix} precision = {precision:.4f}, recall = {recall:.4f}, mean positive score = {pos_rate:.4f}")


def get_binary_classification_details(w, X, y):
    preds, scores = predict_from_weights(w, X)
    y_true = (y.detach().cpu().numpy() == 1).astype(int)
    pred_binary = (preds.detach().cpu().numpy() == 1).astype(int)
    pos_rate = float(scores.detach().cpu().numpy().reshape(-1).mean())
    precision = float(precision_score(y_true, pred_binary, zero_division=0))
    recall = float(recall_score(y_true, pred_binary, zero_division=0))
    return precision, recall, pos_rate


def print_mia_diagnostic(prefix, mia_auc):
    if np.isnan(mia_auc):
        print(f"{prefix} diagnostic: MIA AUC is NaN")
        return
    if mia_auc < 0.5:
        print(f"{prefix} diagnostic: attack direction may be inverted; 1-AUC = {1.0 - mia_auc:.4f}")
    else:
        print(f"{prefix} diagnostic: attack direction is aligned")


def maybe_sample_debug_subgraph(data, debug_sample_size):
    if debug_sample_size <= 0 or debug_sample_size >= data.num_nodes:
        return data

    split_names = ['train_mask', 'val_mask', 'test_mask']
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
        if name in ['x', 'edge_index', 'y']:
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


def _candidate_base_dirs(data_dir):
    script_dir = osp.dirname(osp.abspath(__file__))
    candidates = []

    raw_candidates = [data_dir]
    if not osp.isabs(data_dir):
        raw_candidates.append(osp.join(script_dir, data_dir))

    for candidate in raw_candidates:
        abs_candidate = osp.abspath(candidate)
        if abs_candidate not in candidates:
            candidates.append(abs_candidate)

    project_root = osp.abspath(osp.join(script_dir, '..', '..'))
    for candidate in [
        project_root,
        osp.join(project_root, 'data'),
        osp.join(project_root, 'PyG_datasets'),
        osp.join(project_root, 'PyG_datasets', 'data'),
    ]:
        if candidate not in candidates:
            candidates.append(candidate)

    return candidates


def resolve_planetoid_root(data_dir, dataset_name):
    dataset_name = dataset_name.lower()
    for base_dir in _candidate_base_dirs(data_dir):
        for candidate in [
            osp.join(base_dir, dataset_name),
            osp.join(base_dir, 'data', dataset_name),
        ]:
            raw_dir = osp.join(candidate, dataset_name, 'raw')
            processed_dir = osp.join(candidate, dataset_name, 'processed')
            if osp.isdir(raw_dir) or osp.isdir(processed_dir):
                return candidate

    return osp.join(osp.abspath(data_dir), 'data', dataset_name)


def resolve_amazon_root(data_dir, dataset_name):
    raw_file = f'amazon_electronics_{dataset_name.lower()}.npz'
    for base_dir in _candidate_base_dirs(data_dir):
        for candidate in [
            osp.join(base_dir, 'Amazon'),
            osp.join(base_dir, 'data', 'Amazon'),
            base_dir,
        ]:
            raw_path = osp.join(candidate, dataset_name, 'raw', raw_file)
            processed_path = osp.join(candidate, dataset_name, 'processed', 'data.pt')
            if osp.exists(raw_path) or osp.exists(processed_path):
                return candidate

    return osp.join(osp.abspath(data_dir), 'data', dataset_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a removal-enabled linear model [node/feature]')
    parser.add_argument('--data_dir', type=str, default='./PyG_datasets', help='data directory')
    parser.add_argument('--result_dir', type=str, default='result', help='directory for saving results')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--lam', type=float, default=1e-2, help='L2 regularization')
    parser.add_argument('--std', type=float, default=1e-2, help='standard deviation for objective perturbation')
    parser.add_argument('--num_removes', type=int, default=500, help='number of data points to remove')
    parser.add_argument('--num_steps', type=int, default=100, help='number of optimization steps')
    parser.add_argument('--train_mode', type=str, default='ovr', help='train mode [ovr/binary]')
    parser.add_argument('--train_sep', action='store_true', default=False, help='train binary classifiers separately')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbosity in optimizer')
    # New arguments below
    parser.add_argument('--device', type=int, default=0, help='nonnegative int for cuda id, -1 for cpu')
    parser.add_argument('--prop_step', type=int, default=2, help='number of steps of graph propagation/convolution')
    parser.add_argument('--alpha', type=float, default=0.0, help='we use D^{-a}AD^{-(1-a)} as propagation matrix')
    parser.add_argument('--XdegNorm', type=bool, default=False, help='Apply our degree normaliztion trick')
    parser.add_argument('--add_self_loops', type=bool, default=True, help='Add self loops in propagation matrix')
    parser.add_argument('--optimizer', type=str, default='LBFGS', help='Choice of optimizer. [LBFGS/Adam]')
    parser.add_argument('--lr', type=float, default=1, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay factor for Adam')
    parser.add_argument('--featNorm', type=bool, default=True, help='Row normalize feature to norm 1.')
    parser.add_argument('--GPR', action='store_true', default=False, help='Use GPR model')
    parser.add_argument('--balance_train', action='store_true', default=False, help='Subsample training set to make it balance in class size.')
    parser.add_argument('--Y_binary', type=str, default='0', help='In binary mode, is Y_binary class or Y_binary_1 vs Y_binary_2 (i.e., 0+1).')
    parser.add_argument('--noise_mode', type=str, default='data', help='Data dependent noise or worst case noise [data/worst].')
    parser.add_argument('--removal_mode', type=str, default='node', help='[feature/edge/node].')
    parser.add_argument('--eps', type=float, default=1.0, help='Eps coefficient for certified removal.')
    parser.add_argument('--delta', type=float, default=1e-4, help='Delta coefficient for certified removal.')
    parser.add_argument('--disp', type=int, default=10, help='Display frequency.')
    parser.add_argument('--trails', type=int, default=10, help='Number of repeated trails.')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='Use fixed random seed for removal queue.')
    parser.add_argument('--compare_gnorm', action='store_true', default=False, help='Compute norm of worst case and real gradient each round.')
    parser.add_argument('--compare_retrain', action='store_true', default=False, help='Compare acc with retraining each round.')
    parser.add_argument('--compare_guo', action='store_true', default=False, help='Compare performance with Guo et al.')
    parser.add_argument('--debug_sample_size', type=int, default=0, help='Use an induced subgraph with this many nodes for quick debugging.')
    # Use this if turning into .py code
    args = parser.parse_args()

    # Use this if running using notebook
    # args = parser.parse_args([])

    # this script is only for feature/node removal
    assert args.removal_mode in ['feature', 'node']
    # dont compute norm together with retrain
    assert not (args.compare_gnorm and args.compare_retrain)

    # ========== 核心修复1：安全的设备选择 ==========
    # 检查CUDA是否可用，避免无效设备序号
    if args.device > -1:
        if torch.cuda.is_available():
            # 检查指定的GPU是否存在
            if args.device >= torch.cuda.device_count():
                print(f"警告：指定的GPU ID {args.device} 不存在，自动使用GPU 0")
                device = torch.device("cuda:0")
            else:
                device = torch.device(f"cuda:{args.device}")
            # 设置默认CUDA设备
            torch.cuda.set_device(device)
        else:
            print("警告：CUDA不可用，自动切换到CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    ######
    # Load the data
    print('='*10 + 'Loading data' + '='*10)
    print('Dataset:', args.dataset)
    dataset_name = args.dataset.lower()
    num_classes = None
    # read data from PyG datasets (cora, citeseer, pubmed)
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        path = resolve_planetoid_root(args.data_dir, args.dataset)
        print(f'Using Planetoid root: {path}')
        dataset = Planetoid(path, args.dataset, split="full")
        data = dataset[0]  # 先加载到CPU，再统一移到目标设备
        num_classes = dataset.num_classes
    elif dataset_name in ['ogbn-arxiv', 'ogbn-products']:
        dataset = PygNodePropPredDataset(name=args.dataset, root=args.data_dir)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.val_mask[split_idx['valid']] = True
        data.test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.test_mask[split_idx['test']] = True
        data.y = data.y.squeeze(-1)
        num_classes = dataset.num_classes
    elif dataset_name in ['computers', 'photo']:
        path = resolve_amazon_root(args.data_dir, args.dataset)
        print(f'Using Amazon root: {path}')
        dataset = Amazon(path, args.dataset)
        data = dataset[0]
        data = random_planetoid_splits(data, num_classes=dataset.num_classes, val_lb=500, test_lb=1000, Flag=1)
        num_classes = dataset.num_classes
    elif dataset_name == 'dgraphfin':
        dataset = DGraphFin(root=args.data_dir, name='DGraphFin')  # 使用自定义加载器，指定 name 参数
        data = dataset[0]
        # 规范化 mask 名称：DGraphFin 使用 valid_mask，但代码期望 val_mask
        if hasattr(data, 'valid_mask') and not hasattr(data, 'val_mask'):
            data.val_mask = data.valid_mask
        if data.y.dim() > 1:
            data.y = data.y.squeeze(-1)
        num_classes = int(data.y.max().item()) + 1
    else:
        raise("Error: Not supported dataset yet.")

    if num_classes is None:
        num_classes = int(data.y.max().item()) + 1
    if args.debug_sample_size > 0:
        print(f'Applying debug subgraph sampling with {args.debug_sample_size} nodes')
        data = maybe_sample_debug_subgraph(data, args.debug_sample_size)
        if data.y.dim() > 1:
            data.y = data.y.squeeze(-1)
        num_classes = int(data.y.max().item()) + 1
        print("Debug graph node:{}, edge:{}, train:{}, val:{}, test:{}".format(
            data.num_nodes, data.edge_index.shape[1],
            int(data.train_mask.sum()), int(data.val_mask.sum()), int(data.test_mask.sum())))
    
    # ========== 核心修复2：统一数据设备迁移 ==========
    # 将所有数据统一移到目标设备，避免分步迁移导致的设备不匹配
    data = data.to(device)

    # save the degree of each node for later use
    row = data.edge_index[0]
    deg = degree(row).to(device)  # 确保degree在正确设备上

    # ========== 核心修复3：节点特征处理优化 ==========
    # process features
    if args.featNorm:
        # preprocess_data内部已处理设备问题，直接使用返回结果
        X = preprocess_data(data.x)
    else:
        X = data.x.float()  # 确保数据类型为float
    
    # save a copy of X for removal (确保在正确设备上)
    X_scaled_copy_guo = X.clone().detach().to(device)

    # ========== 核心修复4：标签处理优化 ==========
    # process labels
    if args.train_mode == 'binary':
        if '+' in args.Y_binary:
            # two classes are specified
            class1 = int(args.Y_binary.split('+')[0])
            class2 = int(args.Y_binary.split('+')[1])
            Y = data.y.clone().detach().float()
            Y[data.y == class1] = 1
            Y[data.y == class2] = -1
            interested_data_mask = (data.y == class1) + (data.y == class2)
            train_mask = data.train_mask * interested_data_mask
            val_mask = data.val_mask * interested_data_mask
            test_mask = data.test_mask * interested_data_mask
        else:
            # one vs rest
            class1 = int(args.Y_binary)
            Y = data.y.clone().detach().float()
            Y[data.y == class1] = 1
            Y[data.y != class1] = -1
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask


        # 直接使用mask索引，避免多次设备迁移
        y_train = Y[train_mask]
        y_val = Y[val_mask]
        y_test = Y[test_mask]
    else:
        # multiclass classification
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        
        # one-hot编码并调整范围，确保在正确设备上
        y_train = F.one_hot(data.y[train_mask], num_classes=num_classes) * 2 - 1
        y_train = y_train.float()
        y_val = data.y[val_mask]
        y_test = data.y[test_mask]

    mia_labels = Y if args.train_mode == 'binary' else data.y

    assert args.noise_mode == 'data'

    if args.compare_gnorm:
        # if we want to compare the residual gradient norm of three cases, we should not add noise
        # and make budget very large
        b_std = 0
    else:
        if args.noise_mode == 'data':
            b_std = args.std
        elif args.noise_mode == 'worst':
            b_std = args.std  # change to worst case sigma
        else:
            raise("Error: Not supported noise model.")

    #############
    # initial training with graph
    print('='*10 + 'Training on full dataset with graph' + '='*10)
    start = time.time()
    # 确保传播层在正确设备上
    Propagation = MyGraphConv(K=args.prop_step, add_self_loops=args.add_self_loops,
                              alpha=args.alpha, XdegNorm=args.XdegNorm, GPR=args.GPR).to(device)

    if args.prop_step > 0:
        X = Propagation(X, data.edge_index)

    X = X.float()
    # 直接索引，避免重复to(device)
    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    y_train_labels = data.y[train_mask]
    effective_train_id = torch.arange(data.x.shape[0], device=device)[train_mask]

    print("Train node:{}, Val node:{}, Test node:{}, Edges:{}, Feature dim:{}".format(
        X_train.shape[0], X_val.shape[0], X_test.shape[0],
        data.edge_index.shape[1], X_train.shape[1]))
    if args.train_mode != 'binary':
        print_label_distribution('Train', y_train_labels, num_classes)
        print_label_distribution('Val', y_val, num_classes)
        print_label_distribution('Test', y_test, num_classes)

    if args.balance_train and args.train_mode != 'binary':
        train_balance_mask = get_balance_train_mask(y_train_labels, num_classes).to(X_train.device)
        X_train = X_train[train_balance_mask]
        y_train = y_train[train_balance_mask]
        y_train_labels = y_train_labels[train_balance_mask]
        effective_train_id = effective_train_id[train_balance_mask]
        print("Balanced train node:{}".format(X_train.shape[0]))
        print_label_distribution('Balanced Train', y_train_labels, num_classes)

    ############
    # train removal-enabled linear model
    print("With graph, train mode:", args.train_mode, ", optimizer:", args.optimizer)

    # reserved for future extension
    weight = None
    # in our case weight should always be None
    assert weight is None
    # record the optimal gradient norm wrt the whole training set
    opt_grad_norm = 0

    if args.train_mode == 'ovr':
        # 确保噪声张量在正确设备上
        b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
        if args.train_sep:
            # train K binary LR models separately
            w = torch.zeros(b.size()).float().to(device)
            for k in range(y_train.size(1)):
                if weight is None:
                    w[:, k] = lr_optimize(X_train, y_train[:, k], args.lam, b=b[:, k], 
                                          num_steps=args.num_steps, verbose=args.verbose,
                                          opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                else:
                    w[:, k] = lr_optimize(X_train[weight[:, k].gt(0)], y_train[:, k][weight[:, k].gt(0)], args.lam,
                                          b=b[:, k], num_steps=args.num_steps, verbose=args.verbose,
                                          opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
        else:
            # train K binary LR models jointly
            w = ovr_lr_optimize(X_train, y_train, args.lam, weight, b=b, num_steps=args.num_steps, 
                                verbose=args.verbose, opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
        # record the opt_grad_norm
        for k in range(y_train.size(1)):
            opt_grad_norm += lr_grad(w[:, k], X_train, y_train[:, k], args.lam).norm().cpu()
    else:
        b = b_std * torch.randn(X_train.size(1)).float().to(device)
        w = lr_optimize(X_train, y_train, args.lam, b=b, num_steps=args.num_steps, verbose=args.verbose,
                        opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
        opt_grad_norm = lr_grad(w, X_train, y_train, args.lam).norm().cpu()

    print('Time elapsed: %.2fs' % (time.time() - start))
    train_metrics = evaluate_metrics(w, X_train, y_train_labels if args.train_mode != 'binary' else y_train, args.train_mode)
    val_metrics = evaluate_metrics(w, X_val, y_val, args.train_mode)
    test_metrics = evaluate_metrics(w, X_test, y_test, args.train_mode)
    print_metric_summary('Train', train_metrics)
    print_metric_summary('Val', val_metrics)
    print_metric_summary('Test', test_metrics)
    if args.train_mode != 'binary':
        print_prediction_distribution('Train', w, X_train, num_classes)
        print_prediction_distribution('Val', w, X_val, num_classes)
        print_prediction_distribution('Test', w, X_test, num_classes)
    else:
        print_binary_classification_details('Train', w, X_train, y_train)
        print_binary_classification_details('Val', w, X_val, y_val)
        print_binary_classification_details('Test', w, X_test, y_test)
# 测试遗忘前的 MIA 攻击强度
    mia_auc_before = membership_inference_attack(
        w, X, train_mask, test_mask, y=mia_labels, train_mode=args.train_mode,
        lam=args.lam, shadow_num_steps=args.num_steps, shadow_optimizer=args.optimizer,
        shadow_lr=args.lr, shadow_wd=args.wd, max_samples_per_class=300
    )
    print(f"[MIA Before Unlearning] attack AUC = {mia_auc_before:.4f}")
    print_mia_diagnostic("[MIA Before Unlearning]", mia_auc_before)

    ###########
    if args.compare_guo:
        # initial training without graph
        print('='*10 + 'Training on full dataset without graph' + '='*10)
        start = time.time()

        # only the data preparation part is different
        X_train = X_scaled_copy_guo[effective_train_id]
        X_val = X_scaled_copy_guo[val_mask]
        X_test = X_scaled_copy_guo[test_mask]

        print("Train node:{}, Val node:{}, Test node:{}, Feature dim:{}".format(
            X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1]))
        ######
        # train removal-enabled linear model without graph
        print("Without graph, train mode:", args.train_mode, ", optimizer:", args.optimizer)

        weight = None
        # in our case weight should always be None
        assert weight is None
        opt_grad_norm_guo = 0

        if args.train_mode == 'ovr':
            b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
            if args.train_sep:
                # train K binary LR models separately
                w_guo = torch.zeros(b.size()).float().to(device)
                for k in range(y_train.size(1)):
                    if weight is None:
                        w_guo[:, k] = lr_optimize(X_train, y_train[:, k], args.lam, b=b[:, k], 
                                                  num_steps=args.num_steps, verbose=args.verbose,
                                                  opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                    else:
                        w_guo[:, k] = lr_optimize(X_train[weight[:, k].gt(0)], y_train[:, k][weight[:, k].gt(0)], args.lam,
                                                  b=b[:, k], num_steps=args.num_steps, verbose=args.verbose,
                                                  opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
            else:
                # train K binary LR models jointly
                w_guo = ovr_lr_optimize(X_train, y_train, args.lam, weight, b=b, num_steps=args.num_steps, 
                                        verbose=args.verbose, opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
            # record the opt_grad_norm
            for k in range(y_train.size(1)):
                opt_grad_norm_guo += lr_grad(w_guo[:, k], X_train, y_train[:, k], args.lam).norm().cpu()
        else:
            b = b_std * torch.randn(X_train.size(1)).float().to(device)
            w_guo = lr_optimize(X_train, y_train, args.lam, b=b, num_steps=args.num_steps, verbose=args.verbose, 
                                opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
            opt_grad_norm_guo = lr_grad(w_guo, X_train, y_train, args.lam).norm().cpu()

        print('Time elapsed: %.2fs' % (time.time() - start))
        train_metrics_guo = evaluate_metrics(w_guo, X_train, y_train_labels if args.train_mode != 'binary' else y_train, args.train_mode)
        val_metrics_guo = evaluate_metrics(w_guo, X_val, y_val, args.train_mode)
        test_metrics_guo = evaluate_metrics(w_guo, X_test, y_test, args.train_mode)
        print_metric_summary('Train', train_metrics_guo)
        print_metric_summary('Val', val_metrics_guo)
        print_metric_summary('Test', test_metrics_guo)
        if args.train_mode != 'binary':
            print_prediction_distribution('Train', w_guo, X_train, num_classes)
            print_prediction_distribution('Val', w_guo, X_val, num_classes)
            print_prediction_distribution('Test', w_guo, X_test, num_classes)
        else:
            print_binary_classification_details('Train', w_guo, X_train, y_train)
            print_binary_classification_details('Val', w_guo, X_val, y_val)
            print_binary_classification_details('Test', w_guo, X_test, y_test)

    ###########
    # budget for removal
    c_val = get_c(args.delta)
    # if we need to compute the norms, we should not retrain at all
    if args.compare_gnorm:
        budget = 1e5
    else:
        if args.train_mode == 'ovr':
            budget = get_budget(b_std, args.eps, c_val) * y_train.size(1)
        else:
            budget = get_budget(b_std, args.eps, c_val)
    gamma = 1/4  # pre-computed for -logsigmoid loss
    print('Budget:', budget)

    ##########
    # our removal
    # all norm here is NOT accumulated, need to use np.cumsum in plots
    # ========== 核心修复5：结果张量设备优化 ==========
    # 结果张量默认创建在CPU上（避免占用GPU内存）
    grad_norm_approx = torch.zeros((args.num_removes, args.trails)).float()
    removal_times = torch.zeros((args.num_removes, args.trails)).float()
    acc_removal = torch.zeros((2, args.num_removes, args.trails)).float()
    f1_removal = torch.zeros((2, args.num_removes, args.trails)).float()
    auc_removal = torch.zeros((2, args.num_removes, args.trails)).float()
    precision_removal = torch.full((2, args.num_removes, args.trails), float('nan')).float()
    recall_removal = torch.full((2, args.num_removes, args.trails), float('nan')).float()
    pos_rate_removal = torch.full((2, args.num_removes, args.trails), float('nan')).float()
    grad_norm_worst = torch.zeros((args.num_removes, args.trails)).float()
    grad_norm_real = torch.zeros((args.num_removes, args.trails)).float()
    # graph retrain
    removal_times_graph_retrain = torch.zeros((args.num_removes, args.trails)).float()
    acc_graph_retrain = torch.zeros((2, args.num_removes, args.trails)).float()
    f1_graph_retrain = torch.zeros((2, args.num_removes, args.trails)).float()
    auc_graph_retrain = torch.zeros((2, args.num_removes, args.trails)).float()
    precision_graph_retrain = torch.full((2, args.num_removes, args.trails), float('nan')).float()
    recall_graph_retrain = torch.full((2, args.num_removes, args.trails), float('nan')).float()
    pos_rate_graph_retrain = torch.full((2, args.num_removes, args.trails), float('nan')).float()
    # guo removal
    grad_norm_approx_guo = torch.zeros((args.num_removes, args.trails)).float()
    removal_times_guo = torch.zeros((args.num_removes, args.trails)).float()
    acc_guo = torch.zeros((2, args.num_removes, args.trails)).float()
    f1_guo = torch.zeros((2, args.num_removes, args.trails)).float()
    auc_guo = torch.zeros((2, args.num_removes, args.trails)).float()
    precision_guo = torch.full((2, args.num_removes, args.trails), float('nan')).float()
    recall_guo = torch.full((2, args.num_removes, args.trails), float('nan')).float()
    pos_rate_guo = torch.full((2, args.num_removes, args.trails), float('nan')).float()
    grad_norm_real_guo = torch.zeros((args.num_removes, args.trails)).float()
    # guo retrain
    removal_times_guo_retrain = torch.zeros((args.num_removes, args.trails)).float()
    acc_guo_retrain = torch.zeros((2, args.num_removes, args.trails)).float()
    f1_guo_retrain = torch.zeros((2, args.num_removes, args.trails)).float()
    auc_guo_retrain = torch.zeros((2, args.num_removes, args.trails)).float()
    precision_guo_retrain = torch.full((2, args.num_removes, args.trails), float('nan')).float()
    recall_guo_retrain = torch.full((2, args.num_removes, args.trails), float('nan')).float()
    pos_rate_guo_retrain = torch.full((2, args.num_removes, args.trails), float('nan')).float()
    mia_auc_before_all = torch.full((args.trails,), float('nan')).float()
    mia_auc_after_all = torch.full((args.trails,), float('nan')).float()

    for trail_iter in range(args.trails):
        print('*'*10, trail_iter, '*'*10)
        if args.fix_random_seed:
            # fix the random seed for perm
            np.random.seed(trail_iter)
        
        # ========== 修复6：索引处理优化 ==========
        # 确保索引在正确设备上
        train_id = effective_train_id
        perm = torch.randperm(train_id.shape[0], device=device)
        removal_queue = train_id[perm]
        edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool, device=device)

        X_scaled_copy = X_scaled_copy_guo.clone().detach()
        w_approx = w.clone().detach()
        X_old = X.clone().detach()

        num_retrain = 0
        grad_norm_approx_sum = 0
        # start the removal process
        print('='*10 + 'Testing our removal' + '='*10)
        for i in range(args.num_removes):
            # First, replace removal features with 0 vector
            X_scaled_copy[removal_queue[i]] = 0
            if args.removal_mode == 'node':
                # Then remove the correpsonding edges
                edge_mask[data.edge_index[0] == removal_queue[i]] = False
                edge_mask[data.edge_index[1] == removal_queue[i]] = False
                # make sure we do not remove self-loops
                self_loop_idx = torch.logical_and(
                    data.edge_index[0] == removal_queue[i],
                    data.edge_index[1] == removal_queue[i]
                ).nonzero().squeeze(-1)
                if self_loop_idx.numel() > 0:
                    edge_mask[self_loop_idx] = True

            start = time.time()
            # Get propagated features
            if args.prop_step > 0:
                X_new = Propagation(X_scaled_copy, data.edge_index[:, edge_mask])
            else:
                X_new = X_scaled_copy

            X_val_new = X_new[val_mask]
            X_test_new = X_new[test_mask]

            # note that the removed data point should still not be used in computing K or H
            # removal_queue[(i+1):] are the remaining training idx
            K = get_K_matrix(X_new[removal_queue[(i+1):]])
            spec_norm = sqrt_spectral_norm(K)

            if args.train_mode == 'ovr':
                # removal from all one-vs-rest models
                X_rem = X_new[removal_queue[(i+1):]]
                for k in range(y_train.size(1)):
                    y_rem = y_train[perm[(i+1):], k]
                    H_inv = lr_hessian_inv(w_approx[:, k], X_rem, y_rem, args.lam)
                    # grad_i is the difference
                    grad_old = lr_grad(w_approx[:, k], X_old[removal_queue[i:]], y_train[perm[i:], k], args.lam)
                    grad_new = lr_grad(w_approx[:, k], X_rem, y_rem, args.lam)
                    grad_i = grad_old - grad_new
                    Delta = H_inv.mv(grad_i)
                    Delta_p = X_rem.mv(Delta)
                    # update w here. If beta exceed the budget, w_approx will be retrained
                    w_approx[:, k] += Delta
                    # data dependent norm
                    grad_norm_approx[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                    if args.compare_gnorm:
                        grad_norm_real[i, trail_iter] += lr_grad(w_approx[:, k], X_rem, y_rem, args.lam).norm().cpu()
                        if args.removal_mode == 'node':
                            grad_norm_worst[i, trail_iter] += get_worst_Gbound_node(
                                args.lam, X_rem.shape[0], args.prop_step, deg[removal_queue[i]]).cpu()
                        elif args.removal_mode == 'feature':
                            grad_norm_worst[i, trail_iter] += get_worst_Gbound_feature(
                                args.lam, X_rem.shape[0], deg[removal_queue[i]]).cpu()
                # decide after all classes
                if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                    # retrain the model
                    grad_norm_approx_sum = 0
                    b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
                    w_approx = ovr_lr_optimize(X_rem, y_train[perm[(i+1):]], args.lam, weight, b=b, 
                                               num_steps=args.num_steps, verbose=args.verbose,
                                               opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                    num_retrain += 1
                else:
                    grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                # record acc each round
                val_metrics = evaluate_metrics(w_approx, X_val_new, y_val, args.train_mode)
                test_metrics = evaluate_metrics(w_approx, X_test_new, y_test, args.train_mode)
                acc_removal[0, i, trail_iter], f1_removal[0, i, trail_iter], auc_removal[0, i, trail_iter] = val_metrics
                acc_removal[1, i, trail_iter], f1_removal[1, i, trail_iter], auc_removal[1, i, trail_iter] = test_metrics
                if args.train_mode == 'binary':
                    val_precision, val_recall, val_pos_rate = get_binary_classification_details(w_approx, X_val_new, y_val)
                    test_precision, test_recall, test_pos_rate = get_binary_classification_details(w_approx, X_test_new, y_test)
                    precision_removal[0, i, trail_iter] = val_precision
                    recall_removal[0, i, trail_iter] = val_recall
                    pos_rate_removal[0, i, trail_iter] = val_pos_rate
                    precision_removal[1, i, trail_iter] = test_precision
                    recall_removal[1, i, trail_iter] = test_recall
                    pos_rate_removal[1, i, trail_iter] = test_pos_rate
            else:
                # removal from a single binary logistic regression model
                X_rem = X_new[removal_queue[(i+1):]]
                y_rem = y_train[perm[(i+1):]]
                H_inv = lr_hessian_inv(w_approx, X_rem, y_rem, args.lam)
                # grad_i should be the difference
                grad_old = lr_grad(w_approx, X_old[removal_queue[i:]], y_train[perm[i:]], args.lam)
                grad_new = lr_grad(w_approx, X_rem, y_rem, args.lam)
                grad_i = grad_old - grad_new
                Delta = H_inv.mv(grad_i)
                Delta_p = X_rem.mv(Delta)
                w_approx += Delta
                grad_norm_approx[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                if args.compare_gnorm:
                    grad_norm_real[i, trail_iter] += lr_grad(w_approx, X_rem, y_rem, args.lam).norm().cpu()
                    if args.removal_mode == 'node':
                        grad_norm_worst[i, trail_iter] += get_worst_Gbound_node(
                            args.lam, X_rem.shape[0], args.prop_step, deg[removal_queue[i]]).cpu()
                    elif args.removal_mode == 'feature':
                        grad_norm_worst[i, trail_iter] += get_worst_Gbound_feature(
                            args.lam, X_rem.shape[0], deg[removal_queue[i]]).cpu()

                if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                    # retrain the model
                    grad_norm_approx_sum = 0
                    b = b_std * torch.randn(X_train.size(1)).float().to(device)
                    w_approx = lr_optimize(X_rem, y_rem, args.lam, b=b, num_steps=args.num_steps, 
                                           verbose=args.verbose, opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                    num_retrain += 1
                else:
                    grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                # record acc each round
                val_metrics = evaluate_metrics(w_approx, X_val_new, y_val, args.train_mode)
                test_metrics = evaluate_metrics(w_approx, X_test_new, y_test, args.train_mode)
                acc_removal[0, i, trail_iter], f1_removal[0, i, trail_iter], auc_removal[0, i, trail_iter] = val_metrics
                acc_removal[1, i, trail_iter], f1_removal[1, i, trail_iter], auc_removal[1, i, trail_iter] = test_metrics
                if args.train_mode == 'binary':
                    val_precision, val_recall, val_pos_rate = get_binary_classification_details(w_approx, X_val_new, y_val)
                    test_precision, test_recall, test_pos_rate = get_binary_classification_details(w_approx, X_test_new, y_test)
                    precision_removal[0, i, trail_iter] = val_precision
                    recall_removal[0, i, trail_iter] = val_recall
                    pos_rate_removal[0, i, trail_iter] = val_pos_rate
                    precision_removal[1, i, trail_iter] = test_precision
                    recall_removal[1, i, trail_iter] = test_recall
                    pos_rate_removal[1, i, trail_iter] = test_pos_rate

            removal_times[i, trail_iter] = time.time() - start
            # Remember to replace X_old with X_new
            X_old = X_new.clone().detach()
            if i % args.disp == 0:
                print('Iteration %d: time = %.2fs, number of retrain = %d' % (
                    i+1, removal_times[i, trail_iter], num_retrain))
                print('Val acc = %.4f, F1 = %.4f, AUC = %.4f' % (
                    acc_removal[0, i, trail_iter], f1_removal[0, i, trail_iter], auc_removal[0, i, trail_iter]))
                print('Test acc = %.4f, F1 = %.4f, AUC = %.4f' % (
                    acc_removal[1, i, trail_iter], f1_removal[1, i, trail_iter], auc_removal[1, i, trail_iter]))
                
# ================= 植入黑客：顶会级 MIA 攻防检验 =================
        print("\n" + "="*15 + " 执行遗忘后安全评估 " + "="*15)
        # ✅ 修复：使用完整训练集进行 MIA，而非只用删除的节点
        # MIA 的威胁模型：攻击者通过对比删除前后模型的后验来推断成员
        
        from utils import membership_inference_attack
        
        # 🚨 遗忘前：用完整训练集进行成员推断攻击
        # 限制采样大小以确保影子池充足：最多取 300 个成员和 300 个非成员
        # 这样需要影子池 >= 600，实际可用 708 个 ✓
        # Use the same propagated feature space that the target model was trained on.
        mia_auc_before = membership_inference_attack(
            w, X, train_mask, test_mask, y=mia_labels, train_mode=args.train_mode,
            lam=args.lam, shadow_num_steps=args.num_steps, shadow_optimizer=args.optimizer,
            shadow_lr=args.lr, shadow_wd=args.wd, max_samples_per_class=300
        )
        mia_auc_before_all[trail_iter] = mia_auc_before
        print(f"[MIA Before Unlearning] attack AUC = {mia_auc_before:.4f}")
        print_mia_diagnostic("[MIA Before Unlearning]", mia_auc_before)
        
        # 🛡️ 遗忘后：同样用完整训练集，采样数量相同
        # X_old is updated to the latest propagated feature matrix after the removal loop.
        mia_auc_after = membership_inference_attack(
            w_approx, X_old, train_mask, test_mask, y=mia_labels, train_mode=args.train_mode,
            lam=args.lam, shadow_num_steps=args.num_steps, shadow_optimizer=args.optimizer,
            shadow_lr=args.lr, shadow_wd=args.wd, max_samples_per_class=300
        )
        mia_auc_after_all[trail_iter] = mia_auc_after
        print(f"[MIA After Unlearning] attack AUC = {mia_auc_after:.4f}")
        print_mia_diagnostic("[MIA After Unlearning]", mia_auc_after)
        print("="*50 + "\n")
        # =======================================================================

        # retrain each round with graph
        if args.compare_retrain:
            X_scaled_copy = X_scaled_copy_guo.clone().detach()
            edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool, device=device)
            # start the removal process
            print('='*10 + 'Testing with graph retrain' + '='*10)
            for i in range(args.num_removes):
                # First, replace removal features with 0 vector
                X_scaled_copy[removal_queue[i]] = 0
                # Then remove the correpsonding edges
                if args.removal_mode == 'node':
                    edge_mask[data.edge_index[0] == removal_queue[i]] = False
                    edge_mask[data.edge_index[1] == removal_queue[i]] = False
                    # make sure we do not remove self-loops
                    self_loop_idx = torch.logical_and(
                        data.edge_index[0] == removal_queue[i],
                        data.edge_index[1] == removal_queue[i]
                    ).nonzero().squeeze(-1)
                    if self_loop_idx.numel() > 0:
                        edge_mask[self_loop_idx] = True

                start = time.time()
                # Get propagated features
                if args.prop_step > 0:
                    X_new = Propagation(X_scaled_copy, data.edge_index[:, edge_mask])
                else:
                    X_new = X_scaled_copy

                X_val_new = X_new[val_mask]
                X_test_new = X_new[test_mask]

                if args.train_mode == 'ovr':
                    # removal from all one-vs-rest models
                    X_rem = X_new[removal_queue[(i+1):]]
                    y_rem = y_train[perm[(i+1):]]
                    # retrain the model
                    w_graph_retrain = ovr_lr_optimize(X_rem, y_rem, args.lam, weight, b=None, 
                                                      num_steps=args.num_steps, verbose=args.verbose,
                                                      opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                    val_metrics = evaluate_metrics(w_graph_retrain, X_val_new, y_val, args.train_mode)
                    test_metrics = evaluate_metrics(w_graph_retrain, X_test_new, y_test, args.train_mode)
                    acc_graph_retrain[0, i, trail_iter], f1_graph_retrain[0, i, trail_iter], auc_graph_retrain[0, i, trail_iter] = val_metrics
                    acc_graph_retrain[1, i, trail_iter], f1_graph_retrain[1, i, trail_iter], auc_graph_retrain[1, i, trail_iter] = test_metrics
                    if args.train_mode == 'binary':
                        val_precision, val_recall, val_pos_rate = get_binary_classification_details(w_graph_retrain, X_val_new, y_val)
                        test_precision, test_recall, test_pos_rate = get_binary_classification_details(w_graph_retrain, X_test_new, y_test)
                        precision_graph_retrain[0, i, trail_iter] = val_precision
                        recall_graph_retrain[0, i, trail_iter] = val_recall
                        pos_rate_graph_retrain[0, i, trail_iter] = val_pos_rate
                        precision_graph_retrain[1, i, trail_iter] = test_precision
                        recall_graph_retrain[1, i, trail_iter] = test_recall
                        pos_rate_graph_retrain[1, i, trail_iter] = test_pos_rate
                else:
                    # removal from a single binary logistic regression model
                    X_rem = X_new[removal_queue[(i+1):]]
                    y_rem = y_train[perm[(i+1):]]
                    # retrain the model
                    w_graph_retrain = lr_optimize(X_rem, y_rem, args.lam, b=None, num_steps=args.num_steps, 
                                                  verbose=args.verbose, opt_choice=args.optimizer,
                                                  lr=args.lr, wd=args.wd)
                    val_metrics = evaluate_metrics(w_graph_retrain, X_val_new, y_val, args.train_mode)
                    test_metrics = evaluate_metrics(w_graph_retrain, X_test_new, y_test, args.train_mode)
                    acc_graph_retrain[0, i, trail_iter], f1_graph_retrain[0, i, trail_iter], auc_graph_retrain[0, i, trail_iter] = val_metrics
                    acc_graph_retrain[1, i, trail_iter], f1_graph_retrain[1, i, trail_iter], auc_graph_retrain[1, i, trail_iter] = test_metrics
                    if args.train_mode == 'binary':
                        val_precision, val_recall, val_pos_rate = get_binary_classification_details(w_graph_retrain, X_val_new, y_val)
                        test_precision, test_recall, test_pos_rate = get_binary_classification_details(w_graph_retrain, X_test_new, y_test)
                        precision_graph_retrain[0, i, trail_iter] = val_precision
                        recall_graph_retrain[0, i, trail_iter] = val_recall
                        pos_rate_graph_retrain[0, i, trail_iter] = val_pos_rate
                        precision_graph_retrain[1, i, trail_iter] = test_precision
                        recall_graph_retrain[1, i, trail_iter] = test_recall
                        pos_rate_graph_retrain[1, i, trail_iter] = test_pos_rate

                removal_times_graph_retrain[i, trail_iter] = time.time() - start
                if i % args.disp == 0:
                    print('Iteration %d, time = %.2fs' % (i+1, removal_times_graph_retrain[i, trail_iter]))
                    print('Val acc = %.4f, F1 = %.4f, AUC = %.4f' % (
                        acc_graph_retrain[0, i, trail_iter], f1_graph_retrain[0, i, trail_iter], auc_graph_retrain[0, i, trail_iter]))
                    print('Test acc = %.4f, F1 = %.4f, AUC = %.4f' % (
                        acc_graph_retrain[1, i, trail_iter], f1_graph_retrain[1, i, trail_iter], auc_graph_retrain[1, i, trail_iter]))

        #######
        # guo removal
        if args.compare_guo and args.removal_mode != 'edge':
            w_approx_guo = w_guo.clone().detach()
            num_retrain = 0
            grad_norm_approx_sum_guo = 0
            # prepare the train/val/test sets
            X_train = X_scaled_copy_guo[effective_train_id]
            X_train_perm = X_train[perm]
            y_train_perm = y_train[perm]
            K = get_K_matrix(X_train_perm)
            X_val = X_scaled_copy_guo[val_mask]
            X_test = X_scaled_copy_guo[test_mask]
            # start the removal process
            print('='*10 + 'Testing Guo et al. removal' + '='*10)
            for i in range(args.num_removes):
                start = time.time()
                if args.train_mode == 'ovr':
                    # removal from all one-vs-rest models
                    X_rem = X_train_perm[(i+1):]
                    # update matrix K
                    K -= torch.outer(X_train_perm[i], X_train_perm[i])
                    spec_norm = sqrt_spectral_norm(K)
                    for k in range(y_train_perm.size(1)):
                        y_rem = y_train_perm[(i+1):, k]
                        H_inv = lr_hessian_inv(w_approx_guo[:, k], X_rem, y_rem, args.lam)
                        # grad_i is the difference
                        grad_i = lr_grad(w_approx_guo[:, k], X_train_perm[i].unsqueeze(0), 
                                         y_train_perm[i, k].unsqueeze(0), args.lam)
                        Delta = H_inv.mv(grad_i)
                        Delta_p = X_rem.mv(Delta)
                        # update w here. If beta exceed the budget, w_approx_guo will be retrained
                        w_approx_guo[:, k] += Delta
                        grad_norm_approx_guo[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                        if args.compare_gnorm:
                            grad_norm_real_guo[i, trail_iter] += lr_grad(w_approx_guo[:, k], X_rem, y_rem, args.lam).norm().cpu()
                    # decide after all classes
                    if grad_norm_approx_sum_guo + grad_norm_approx_guo[i, trail_iter] > budget:
                        # retrain the model
                        grad_norm_approx_sum_guo = 0
                        b = b_std * torch.randn(X_train_perm.size(1), y_train_perm.size(1)).float().to(device)
                        w_approx_guo = ovr_lr_optimize(X_rem, y_train_perm[(i+1):], args.lam, weight, b=b, 
                                                       num_steps=args.num_steps, verbose=args.verbose,
                                                       opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                        num_retrain += 1
                    else:
                        grad_norm_approx_sum_guo += grad_norm_approx_guo[i, trail_iter]
                    # record the acc each round
                    val_metrics = evaluate_metrics(w_approx_guo, X_val, y_val, args.train_mode)
                    test_metrics = evaluate_metrics(w_approx_guo, X_test, y_test, args.train_mode)
                    acc_guo[0, i, trail_iter], f1_guo[0, i, trail_iter], auc_guo[0, i, trail_iter] = val_metrics
                    acc_guo[1, i, trail_iter], f1_guo[1, i, trail_iter], auc_guo[1, i, trail_iter] = test_metrics
                    if args.train_mode == 'binary':
                        val_precision, val_recall, val_pos_rate = get_binary_classification_details(w_approx_guo, X_val, y_val)
                        test_precision, test_recall, test_pos_rate = get_binary_classification_details(w_approx_guo, X_test, y_test)
                        precision_guo[0, i, trail_iter] = val_precision
                        recall_guo[0, i, trail_iter] = val_recall
                        pos_rate_guo[0, i, trail_iter] = val_pos_rate
                        precision_guo[1, i, trail_iter] = test_precision
                        recall_guo[1, i, trail_iter] = test_recall
                        pos_rate_guo[1, i, trail_iter] = test_pos_rate
                else:
                    # removal from a single binary logistic regression model
                    X_rem = X_train_perm[(i+1):]
                    y_rem = y_train_perm[(i+1):]
                    H_inv = lr_hessian_inv(w_approx_guo, X_rem, y_rem, args.lam)
                    grad_i = lr_grad(w_approx_guo, X_train_perm[i].unsqueeze(0), 
                                     y_train_perm[i].unsqueeze(0), args.lam)
                    K -= torch.outer(X_train_perm[i], X_train_perm[i])
                    spec_norm = sqrt_spectral_norm(K)
                    Delta = H_inv.mv(grad_i)
                    Delta_p = X_rem.mv(Delta)
                    w_approx_guo += Delta
                    grad_norm_approx_guo[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                    if args.compare_gnorm:
                        grad_norm_real_guo[i, trail_iter] += lr_grad(w_approx_guo, X_rem, y_rem, args.lam).norm().cpu()
                    if grad_norm_approx_sum_guo + grad_norm_approx_guo[i, trail_iter] > budget:
                        # retrain the model
                        grad_norm_approx_sum_guo = 0
                        b = b_std * torch.randn(X_train_perm.size(1)).float().to(device)
                        w_approx_guo = lr_optimize(X_rem, y_rem, args.lam, b=b, num_steps=args.num_steps, 
                                                   verbose=args.verbose, opt_choice=args.optimizer,
                                                   lr=args.lr, wd=args.wd)
                        num_retrain += 1
                    else:
                        grad_norm_approx_sum_guo += grad_norm_approx_guo[i, trail_iter]
                    # record the acc each round
                    val_metrics = evaluate_metrics(w_approx_guo, X_val, y_val, args.train_mode)
                    test_metrics = evaluate_metrics(w_approx_guo, X_test, y_test, args.train_mode)
                    acc_guo[0, i, trail_iter], f1_guo[0, i, trail_iter], auc_guo[0, i, trail_iter] = val_metrics
                    acc_guo[1, i, trail_iter], f1_guo[1, i, trail_iter], auc_guo[1, i, trail_iter] = test_metrics
                    if args.train_mode == 'binary':
                        val_precision, val_recall, val_pos_rate = get_binary_classification_details(w_approx_guo, X_val, y_val)
                        test_precision, test_recall, test_pos_rate = get_binary_classification_details(w_approx_guo, X_test, y_test)
                        precision_guo[0, i, trail_iter] = val_precision
                        recall_guo[0, i, trail_iter] = val_recall
                        pos_rate_guo[0, i, trail_iter] = val_pos_rate
                        precision_guo[1, i, trail_iter] = test_precision
                        recall_guo[1, i, trail_iter] = test_recall
                        pos_rate_guo[1, i, trail_iter] = test_pos_rate

                removal_times_guo[i, trail_iter] = time.time() - start
                if i % args.disp == 0:
                    print('Iteration %d: time = %.2fs, number of retrain = %d' % (
                        i+1, removal_times_guo[i, trail_iter], num_retrain))
                    print('Val acc = %.4f, F1 = %.4f, AUC = %.4f' % (
                        acc_guo[0, i, trail_iter], f1_guo[0, i, trail_iter], auc_guo[0, i, trail_iter]))
                    print('Test acc = %.4f, F1 = %.4f, AUC = %.4f' % (
                        acc_guo[1, i, trail_iter], f1_guo[1, i, trail_iter], auc_guo[1, i, trail_iter]))

        #######
        # retrain each round without graph
        if args.removal_mode != 'edge' and args.compare_retrain and args.compare_guo:
            X_train = X_scaled_copy_guo[effective_train_id]
            X_train_perm = X_train[perm]
            y_train_perm = y_train[perm]
            X_val = X_scaled_copy_guo[val_mask]
            X_test = X_scaled_copy_guo[test_mask]

            # start the removal process
            print('='*10 + 'Testing without graph retrain' + '='*10)
            for i in range(args.num_removes):
                start = time.time()
                if args.train_mode == 'ovr':
                    # removal from all one-vs-rest models
                    X_rem = X_train_perm[(i+1):]
                    y_rem = y_train_perm[(i+1):]
                    # retrain the model
                    w_guo_retrain = ovr_lr_optimize(X_rem, y_rem, args.lam, weight, b=None, 
                                                    num_steps=args.num_steps, verbose=args.verbose,
                                                    opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                    val_metrics = evaluate_metrics(w_guo_retrain, X_val, y_val, args.train_mode)
                    test_metrics = evaluate_metrics(w_guo_retrain, X_test, y_test, args.train_mode)
                    acc_guo_retrain[0, i, trail_iter], f1_guo_retrain[0, i, trail_iter], auc_guo_retrain[0, i, trail_iter] = val_metrics
                    acc_guo_retrain[1, i, trail_iter], f1_guo_retrain[1, i, trail_iter], auc_guo_retrain[1, i, trail_iter] = test_metrics
                    if args.train_mode == 'binary':
                        val_precision, val_recall, val_pos_rate = get_binary_classification_details(w_guo_retrain, X_val, y_val)
                        test_precision, test_recall, test_pos_rate = get_binary_classification_details(w_guo_retrain, X_test, y_test)
                        precision_guo_retrain[0, i, trail_iter] = val_precision
                        recall_guo_retrain[0, i, trail_iter] = val_recall
                        pos_rate_guo_retrain[0, i, trail_iter] = val_pos_rate
                        precision_guo_retrain[1, i, trail_iter] = test_precision
                        recall_guo_retrain[1, i, trail_iter] = test_recall
                        pos_rate_guo_retrain[1, i, trail_iter] = test_pos_rate
                else:
                    # removal from a single binary logistic regression model
                    X_rem = X_train_perm[(i+1):]
                    y_rem = y_train_perm[(i+1):]
                    # retrain the model
                    w_guo_retrain = lr_optimize(X_rem, y_rem, args.lam, b=None, num_steps=args.num_steps, 
                                                verbose=args.verbose, opt_choice=args.optimizer,
                                                lr=args.lr, wd=args.wd)
                    val_metrics = evaluate_metrics(w_guo_retrain, X_val, y_val, args.train_mode)
                    test_metrics = evaluate_metrics(w_guo_retrain, X_test, y_test, args.train_mode)
                    acc_guo_retrain[0, i, trail_iter], f1_guo_retrain[0, i, trail_iter], auc_guo_retrain[0, i, trail_iter] = val_metrics
                    acc_guo_retrain[1, i, trail_iter], f1_guo_retrain[1, i, trail_iter], auc_guo_retrain[1, i, trail_iter] = test_metrics
                    if args.train_mode == 'binary':
                        val_precision, val_recall, val_pos_rate = get_binary_classification_details(w_guo_retrain, X_val, y_val)
                        test_precision, test_recall, test_pos_rate = get_binary_classification_details(w_guo_retrain, X_test, y_test)
                        precision_guo_retrain[0, i, trail_iter] = val_precision
                        recall_guo_retrain[0, i, trail_iter] = val_recall
                        pos_rate_guo_retrain[0, i, trail_iter] = val_pos_rate
                        precision_guo_retrain[1, i, trail_iter] = test_precision
                        recall_guo_retrain[1, i, trail_iter] = test_recall
                        pos_rate_guo_retrain[1, i, trail_iter] = test_pos_rate

                removal_times_guo_retrain[i, trail_iter] = time.time() - start
                if i % args.disp == 0:
                    print('Iteration %d, time = %.2fs' % (i+1, removal_times_guo_retrain[i, trail_iter]))
                    print('Val acc = %.4f, F1 = %.4f, AUC = %.4f' % (
                        acc_guo_retrain[0, i, trail_iter], f1_guo_retrain[0, i, trail_iter], auc_guo_retrain[0, i, trail_iter]))
                    print('Test acc = %.4f, F1 = %.4f, AUC = %.4f' % (
                        acc_guo_retrain[1, i, trail_iter], f1_guo_retrain[1, i, trail_iter], auc_guo_retrain[1, i, trail_iter]))

    # save all results
    if not osp.exists(args.result_dir):
        os.makedirs(args.result_dir)
    save_path = '%s/%s_std_%.0e_lam_%.0e_nr_%d_K_%d_opt_%s_mode_%s_eps_%.1f_delta_%.0e' % (
        args.result_dir, args.dataset, b_std, args.lam, args.num_removes, args.prop_step, 
        args.optimizer, args.removal_mode, args.eps, args.delta)
    if args.train_mode == 'binary':
        save_path += '_bin_%s' % args.Y_binary
    if args.GPR:
        save_path += '_gpr'
    if args.compare_gnorm:
        save_path += '_gnorm'
    if args.compare_retrain:
        save_path += '_retrain'
    if args.compare_guo:
        save_path += '_withguo'

    save_path += '.pth'
    torch.save({
        'grad_norm_approx': grad_norm_approx, 
        'removal_times': removal_times, 
        'acc_removal': acc_removal,
        'f1_removal': f1_removal,
        'auc_removal': auc_removal,
        'precision_removal': precision_removal,
        'recall_removal': recall_removal,
        'pos_rate_removal': pos_rate_removal,
        'grad_norm_worst': grad_norm_worst, 
        'grad_norm_real': grad_norm_real,
        'mia_auc_before_all': mia_auc_before_all,
        'mia_auc_after_all': mia_auc_after_all,
        'removal_times_graph_retrain': removal_times_graph_retrain, 
        'acc_graph_retrain': acc_graph_retrain,
        'f1_graph_retrain': f1_graph_retrain,
        'auc_graph_retrain': auc_graph_retrain,
        'precision_graph_retrain': precision_graph_retrain,
        'recall_graph_retrain': recall_graph_retrain,
        'pos_rate_graph_retrain': pos_rate_graph_retrain,
        'grad_norm_approx_guo': grad_norm_approx_guo, 
        'removal_times_guo': removal_times_guo, 
        'acc_guo': acc_guo,
        'f1_guo': f1_guo,
        'auc_guo': auc_guo,
        'precision_guo': precision_guo,
        'recall_guo': recall_guo,
        'pos_rate_guo': pos_rate_guo,
        'removal_times_guo_retrain': removal_times_guo_retrain, 
        'acc_guo_retrain': acc_guo_retrain,
        'f1_guo_retrain': f1_guo_retrain,
        'auc_guo_retrain': auc_guo_retrain,
        'precision_guo_retrain': precision_guo_retrain,
        'recall_guo_retrain': recall_guo_retrain,
        'pos_rate_guo_retrain': pos_rate_guo_retrain,
        'grad_norm_real_guo': grad_norm_real_guo
    }, save_path)
