import math
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from sklearn import preprocessing
from numpy.linalg import norm

# Below is for graph learning part
from torch_geometric.nn.conv import MessagePassing
from typing import Optional
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# loss for binary classification
def lr_loss(w, X, y, lam):
    '''
    input: 
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
    return:
        averaged training loss with L2 regularization
    '''
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2


# evaluate function for binary classification
def lr_eval(w, X, y):
    '''
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
    return:
        prediction accuracy
    '''
    return X.mv(w).sign().eq(y).float().mean()


# gradient of loss wrt w for binary classification
def lr_grad(w, X, y, lam):
    '''
    The gradient here is computed wrt sum.
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
    return:
        gradient: (d,)
    '''
    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z-1) * y) + lam * X.size(0) * w


# hessian of loss wrt w for binary classification
def lr_hessian_inv(w, X, y, lam, batch_size=50000, device='cpu'):
    '''
    The hessian here is computed wrt sum.
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
        batch_size: int
    return:
        hessian: (d,d)
    '''
    # 核心修复1：确保所有张量在同一设备
    z = torch.sigmoid(y * X.mv(w)).to(X.device)
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, X.size(0))
        X_i = X[lower:upper].to(X.device)
        D_i = D[lower:upper].unsqueeze(1).to(X.device)
        
        if H is None:
            H = X_i.t().mm(D_i * X_i)
        else:
            H += X_i.t().mm(D_i * X_i)
    
    # 核心修复2：单位矩阵使用X的设备，而非传入的device（避免参数传递错误）
    eye_matrix = torch.eye(X.size(1)).float().to(X.device)
    return (H + lam * X.size(0) * eye_matrix).inverse()


# training iteration for binary classification
def lr_optimize(X, y, lam, b=None, num_steps=100, tol=1e-32, verbose=False, opt_choice='LBFGS', lr=0.01, wd=0, X_val=None, y_val=None, device='cpu'):
    '''
        b is the noise here. It is either pre-computed for worst-case, or pre-defined.
    '''
    # 修复：使用X的设备作为基准，而非传入的device
    actual_device = X.device if isinstance(X, torch.Tensor) else torch.device(device)
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(actual_device), requires_grad=True)
    
    def closure():
        optimizer.zero_grad()  # 修复：closure中添加zero_grad避免梯度累积
        if b is None:
            loss = lr_loss(w, X, y, lam)
        else:
            loss = lr_loss(w, X, y, lam) + b.to(w.device).dot(w) / X.size(0)
        # 修复：消除LBFGS的requires_grad警告
        loss.backward()
        return loss.detach() if loss.requires_grad else loss
    
    if opt_choice == 'LBFGS':
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == 'Adam':
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise("Error: Not supported optimizer.")
    
    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.to(w.device).dot(w) / X.size(0)
        loss.backward()
        
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        
        if opt_choice == 'LBFGS':
            optimizer.step(closure)
        elif opt_choice == 'Adam':
            optimizer.step()
        else:
            raise("Error: Not supported optimizer.")
        
        # If we want to control the norm of w_best, we should keep the last w instead of the one with
        # the highest val acc
        if X_val is not None:
            val_acc = lr_eval(w, X_val, y_val)
            if verbose:
                print('Val accuracy = %.4f' % val_acc, 'Best Val acc = %.4f' % best_val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                w_best = w.clone().detach()
        else:
            w_best = w.clone().detach()

    if w_best is None:
        raise("Error: Training procedure failed")
    return w_best


# aggregated loss for multiclass classification
def ovr_lr_loss(w, X, y, lam, weight=None):
    '''
     input:
        w: (d,c)
        X: (n,d)
        y: (n,c), one-hot
        lambda: scalar
        weight: (c,) / None
    return:
        loss: scalar
    '''
    z = batch_multiply(X, w) * y
    if weight is None:
        return -F.logsigmoid(z).mean(0).sum() + lam * w.pow(2).sum() / 2
    else:
        return -F.logsigmoid(z).mul_(weight).sum() + lam * w.pow(2).sum() / 2


def ovr_lr_eval(w, X, y):
    '''
    input:
        w: (d,c)
        X: (n,d)
        y: (n,), NOT one-hot
    return:
        loss: scalar
    '''
    pred = X.mm(w).max(1)[1]
    return pred.eq(y).float().mean()


def ovr_lr_optimize(X, y, lam, weight=None, b=None, num_steps=100, tol=1e-32, verbose=False, opt_choice='LBFGS', lr=0.01, wd=0, X_val=None, y_val=None, device='cpu'):
    '''
    y: (n_train, c). one-hot
    y_val: (n_val,) NOT one-hot
    '''
    # 修复：使用X的设备作为基准
    actual_device = X.device if isinstance(X, torch.Tensor) else torch.device(device)
    
    # We use random initialization as in common DL literature.
    # w = torch.zeros(X.size(1), y.size(1)).float()
    # init.kaiming_uniform_(w, a=math.sqrt(5))
    # w = torch.autograd.Variable(w.to(device), requires_grad=True)
    # zero initialization
    w = torch.autograd.Variable(torch.zeros(X.size(1), y.size(1)).float().to(actual_device), requires_grad=True)

    def closure():
        optimizer.zero_grad()  # 修复：closure中添加zero_grad
        if b is None:
            loss = ovr_lr_loss(w, X, y, lam, weight)
        else:
            loss = ovr_lr_loss(w, X, y, lam, weight) + (b.to(w.device) * w).sum() / X.size(0)
        loss.backward()  # 修复：消除LBFGS的requires_grad警告
        return loss.detach() if loss.requires_grad else loss
    
    if opt_choice == 'LBFGS':
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == 'Adam':
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise("Error: Not supported optimizer.")
    
    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = ovr_lr_loss(w, X, y, lam, weight)
        if b is not None:
            if weight is None:
                loss += (b.to(w.device) * w).sum() / X.size(0)
            else:
                loss += ((b.to(w.device) * w).sum(0) * weight.max(0)[0]).sum()
        loss.backward()
        
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        
        if opt_choice == 'LBFGS':
            optimizer.step(closure)
        elif opt_choice == 'Adam':
            optimizer.step()
        else:
            raise("Error: Not supported optimizer.")
        
        if X_val is not None:
            val_acc = ovr_lr_eval(w, X_val, y_val)
            if verbose:
                print('Val accuracy = %.4f' % val_acc, 'Best Val acc = %.4f' % best_val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                w_best = w.clone().detach()
        else:
            w_best = w.clone().detach()
    
    if w_best is None:
        raise("Error: Training procedure failed")
    return w_best


def batch_multiply(A, B, batch_size=500000, device='cpu'):
    # 核心修复：统一使用A/B的设备，而非传入的device
    if A.is_cuda or B.is_cuda:
        target_device = A.device if A.is_cuda else B.device
        A = A.to(target_device)
        B = B.to(target_device)
        if len(B.size()) == 1:
            return A.mv(B)
        else:
            return A.mm(B)
    else:
        out = []
        num_batch = int(math.ceil(A.size(0) / float(batch_size)))
        with torch.no_grad():
            for i in range(num_batch):
                lower = i * batch_size
                upper = min((i+1) * batch_size, A.size(0))
                A_sub = A[lower:upper].to(B.device)
                if len(B.size()) == 1:
                    out.append(A_sub.mv(B).cpu())
                else:
                    out.append(A_sub.mm(B).cpu())
        # 修复：最终返回和A同设备
        return torch.cat(out, dim=0).to(A.device)


# using power iteration to find the maximum eigenvalue
def sqrt_spectral_norm(A, num_iters=100, device='cpu'):
    '''
    return:
        sqrt of maximum eigenvalue/spectral norm
    device: torch.device, default 'cpu'
    '''
    # 核心修复：x的设备和A保持一致，优先用A的设备
    x = torch.randn(A.size(0)).float().to(A.device)
    for i in range(num_iters):
        x = A.mv(x)
        x_norm = x.norm()
        x /= x_norm
    max_lam = torch.dot(x, A.mv(x)) / torch.dot(x, x)
    return math.sqrt(max_lam)


# prepare P matrix in PyG format
def get_propagation(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None, alpha=0.5):
    """
    return:
        P = D^{-\alpha}AD^{-(1-alpha)}.
    """
    fill_value = 2. if improved else 1.
    assert (0 <= alpha) and (alpha <= 1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    # 修复：确保edge_weight和edge_index同设备
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
    else:
        edge_weight = edge_weight.to(edge_index.device)
        
    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_left = deg.pow(-alpha)
    deg_inv_right = deg.pow(alpha-1)
    deg_inv_left.masked_fill_(deg_inv_left == float('inf'), 0)
    deg_inv_right.masked_fill_(deg_inv_right == float('inf'), 0)

    return edge_index, deg_inv_left[row] * edge_weight * deg_inv_right[col]


class MyGraphConv(MessagePassing):
    """
    Use customized propagation matrix. Just PX (or PD^{-1}X), no linear layer yet.
    """
    _cached_x: Optional[Tensor]

    def __init__(self, K: int = 1,
                 add_self_loops: bool = True,
                 alpha=0.5, XdegNorm=False, GPR=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.K = K
        self.add_self_loops = add_self_loops
        self.alpha = alpha
        self.XdegNorm = XdegNorm
        self.GPR = GPR
        self._cached_x = None # Not used
        self.reset_parameters()

    def reset_parameters(self):
        self._cached_x = None # Not used

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        # 修复：确保输入张量同设备
        if isinstance(edge_index, Tensor):
            edge_index = edge_index.to(x.device)
            edge_index, edge_weight = get_propagation(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype, alpha=self.alpha)
        elif isinstance(edge_index, SparseTensor):
            edge_index = get_propagation(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype, alpha=self.alpha)
        
        if self.XdegNorm:
            # X <-- D^{-1}X, our degree normalization trick
            num_nodes = maybe_num_nodes(edge_index, None)
            row, col = edge_index[0], edge_index[1]
            deg = degree(row).unsqueeze(-1).to(x.device)
            
            deg_inv = deg.pow(-1)
            deg_inv = deg_inv.masked_fill_(deg_inv == float('inf'), 0) 
        
        if self.GPR:
            xs = []
            xs.append(x)
            if self.XdegNorm:
                x = deg_inv * x # X <-- D^{-1}X
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
                xs.append(x)
            return torch.cat(xs, dim=1) / (self.K + 1)
        else:
            if self.XdegNorm:
                x = deg_inv * x # X <-- D^{-1}X
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')


# K = X^T * X for fast computation of spectral norm
def get_K_matrix(X):
    # 修复：确保矩阵乘法在同一设备
    X = X.to(X.device)
    K = X.t().mm(X)
    return K


def index_to_mask(index, size):
    # 修复：确保mask和index同设备
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, test_lb=1000, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    if Flag == 0:
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)
    else:
        all_index = torch.randperm(data.y.shape[0]).to(data.y.device)
        data.val_mask = index_to_mask(all_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(all_index[val_lb: (val_lb+test_lb)], size=data.num_nodes)
        data.train_mask = index_to_mask(all_index[(val_lb+test_lb):], size=data.num_nodes)
    return data


def get_balance_train_mask(y_train, num_classes):
    """
    Make the size of each class in training set = the smallest class.
    """
    # Find the smallest class size
    C_size = torch.zeros(num_classes)
    for i in range(num_classes):
        C_size[i] = (y_train == i).sum()
        
    C_size_exceed = C_size - C_size.min()
    
    # For each class, remove nodes such that size = C_min.
    train_balance_mask = torch.ones(y_train.shape[0]).to(y_train.device)
    All_train_id = np.arange(y_train.shape[0])
    for i in range(num_classes):
        if int(C_size_exceed[i])>0:
            pick = np.random.choice(All_train_id[y_train==i],int(C_size_exceed[i]),replace=False)
            train_balance_mask[pick] = 0
    return train_balance_mask.type(torch.BoolTensor)


def preprocess_data(X):
    '''
    input:
        X: (n,d), torch.Tensor
    '''
    # 修复：处理CPU/GPU张量
    X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
    scaler = preprocessing.StandardScaler().fit(X_np)
    X_scaled = scaler.transform(X_np)
    row_norm = norm(X_scaled, axis=1)
    X_scaled = X_scaled / row_norm.max()
    
    # 修复：返回和原X同设备的张量
    return torch.from_numpy(X_scaled).to(X.device) if isinstance(X, torch.Tensor) else torch.from_numpy(X_scaled)


def get_worst_Gbound_feature(lam, m, deg_m, gamma1=0.25, gamma2=0.25, c=1, c1=1):
    return gamma2 * ((2*c*lam + (c*gamma1+lam*c1)*deg_m) ** 2) / (lam ** 4) / (m-1)


def get_worst_Gbound_edge(lam, m, K, gamma1=0.25, gamma2=0.25, c=1, c1=1):
    return 16 * gamma2 * (K**2) * ((c*gamma1+lam*c1) ** 2) / (lam ** 4) / m


def get_worst_Gbound_node(lam, m, K, deg_m, gamma1=0.25, gamma2=0.25, c=1, c1=1):
    return gamma2 * ((2*c*lam + K*(c*gamma1+lam*c1)*(2*deg_m-1)) ** 2) / (lam ** 4) / (m-1)


def get_c(delta):
    return np.sqrt(2*np.log(1.5/delta))


def get_budget(std, eps, c):
    return std * eps / c

def deprecated_membership_inference_attack_v1(w, X, train_mask, test_mask):
    """
    轻量级成员推断攻击 (MIA)
    目标: 区分样本是来自训练集 (Members, 标签为1) 还是测试集 (Non-members, 标签为0)
    """
    # 1. 获取目标模型对所有数据的预测置信度 (即 logits 经过 sigmoid)
    with torch.no_grad():
        logits = X.mv(w)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        # 为了让黑客更容易学，我们提取预测的"熵"或绝对置信度
        # 置信度越高(越接近0或1)，越有可能是训练集
        confidence = np.abs(probs - 0.5) * 2  # 映射到 0~1 的绝对置信度
        
    # 2. 准备黑客的训练数据
    # 提取训练集(见过的)和测试集(没见过的)的置信度
    members_conf = confidence[train_mask.cpu().numpy()]
    non_members_conf = confidence[test_mask.cpu().numpy()]
    
    # 拼接数据，构建 MIA 数据集
    # X_mia: 置信度特征; Y_mia: 1表示在训练集中，0表示不在
    X_mia = np.concatenate([members_conf, non_members_conf]).reshape(-1, 1)
    Y_mia = np.concatenate([np.ones(len(members_conf)), np.zeros(len(non_members_conf))])
    
    # 3. 训练黑客模型 (用逻辑回归作为攻击分类器)
    attack_model = LogisticRegression(class_weight='balanced')
    attack_model.fit(X_mia, Y_mia)
    
    # 4. 评估攻击成功率 (使用 AUC 曲线下面积，0.5代表瞎猜，越接近1代表隐私泄露越严重)
    mia_preds = attack_model.predict_proba(X_mia)[:, 1]
    mia_auc = roc_auc_score(Y_mia, mia_preds)
    
    return mia_auc


def deprecated_membership_inference_attack_v2(w, X, train_mask, test_mask):
    """
    Lightweight membership inference attack using prediction confidence.
    Supports both binary linear models and one-vs-rest multiclass weights.
    """
    with torch.no_grad():
        if w.dim() == 1:
            logits = X.mv(w)
            probs = torch.sigmoid(logits)
            confidence = (probs - 0.5).abs().mul(2).cpu().numpy()
        else:
            logits = X.mm(w)
            probs = torch.softmax(logits, dim=1)
            confidence = probs.max(dim=1).values.cpu().numpy()

    members_conf = confidence[train_mask.cpu().numpy()]
    non_members_conf = confidence[test_mask.cpu().numpy()]

    X_mia = np.concatenate([members_conf, non_members_conf]).reshape(-1, 1)
    Y_mia = np.concatenate([np.ones(len(members_conf)), np.zeros(len(non_members_conf))])

    attack_model = LogisticRegression(class_weight='balanced')
    attack_model.fit(X_mia, Y_mia)

    mia_preds = attack_model.predict_proba(X_mia)[:, 1]
    mia_auc = roc_auc_score(Y_mia, mia_preds)

    return mia_auc


def deprecated_membership_inference_attack_v3(w, X, train_mask, test_mask, y=None, max_samples_per_class=50000, random_state=0):
    """
    Stronger MIA baseline with richer attack features and a held-out attack test split.
    """
    with torch.no_grad():
        if w.dim() == 1:
            logits_raw = X.mv(w)
            probs_pos = torch.sigmoid(logits_raw)
            probs = torch.stack([1 - probs_pos, probs_pos], dim=1)
            logits = torch.stack([-logits_raw, logits_raw], dim=1)
        else:
            logits = X.mm(w)
            probs = torch.softmax(logits, dim=1)

        topk = min(2, probs.size(1))
        top_probs = torch.topk(probs, k=topk, dim=1).values
        max_conf = top_probs[:, 0]
        margin = top_probs[:, 0] - top_probs[:, 1] if topk > 1 else top_probs[:, 0]
        entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1)
        if probs.size(1) > 1:
            entropy = entropy / np.log(probs.size(1))

        feature_list = [
            max_conf.unsqueeze(1),
            margin.unsqueeze(1),
            entropy.unsqueeze(1),
        ]

        if y is not None:
            if w.dim() == 1:
                y_idx = (y > 0).long()
            else:
                y_idx = y.long()
            row_idx = torch.arange(probs.size(0), device=probs.device)
            true_prob = probs[row_idx, y_idx].clamp(min=1e-12)
            true_logit = logits[row_idx, y_idx]
            loss = -torch.log(true_prob)
            feature_list.extend([
                true_prob.unsqueeze(1),
                true_logit.unsqueeze(1),
                loss.unsqueeze(1),
            ])

        attack_features = torch.cat(feature_list, dim=1).cpu().numpy()

    member_idx = train_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    non_member_idx = test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()

    rng = np.random.default_rng(random_state)
    if max_samples_per_class is not None:
        if member_idx.shape[0] > max_samples_per_class:
            member_idx = rng.choice(member_idx, size=max_samples_per_class, replace=False)
        if non_member_idx.shape[0] > max_samples_per_class:
            non_member_idx = rng.choice(non_member_idx, size=max_samples_per_class, replace=False)

    sample_count = min(member_idx.shape[0], non_member_idx.shape[0])
    if sample_count < 4:
        return 0.5

    if member_idx.shape[0] > sample_count:
        member_idx = rng.choice(member_idx, size=sample_count, replace=False)
    if non_member_idx.shape[0] > sample_count:
        non_member_idx = rng.choice(non_member_idx, size=sample_count, replace=False)

    X_mia = np.concatenate([attack_features[member_idx], attack_features[non_member_idx]], axis=0)
    Y_mia = np.concatenate([np.ones(sample_count), np.zeros(sample_count)])

    X_train_attack, X_test_attack, y_train_attack, y_test_attack = train_test_split(
        X_mia, Y_mia, test_size=0.5, stratify=Y_mia, random_state=random_state
    )

    attack_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state)
    attack_model.fit(X_train_attack, y_train_attack)

    mia_preds = attack_model.predict_proba(X_test_attack)[:, 1]
    mia_auc = roc_auc_score(y_test_attack, mia_preds)

    return mia_auc


def _posterior_from_linear_model(w, X):
    with torch.no_grad():
        if w.dim() == 1:
            logits_raw = X.mv(w)
            probs_pos = torch.sigmoid(logits_raw)
            probs = torch.stack([1 - probs_pos, probs_pos], dim=1)
        else:
            logits = X.mm(w)
            probs = torch.softmax(logits, dim=1)
    return probs


def _sample_member_nonmember_indices(train_mask, test_mask, max_samples_per_class, random_state):
    member_idx = train_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    non_member_idx = test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    rng = np.random.default_rng(random_state)
    
    print(f"[MIA DIAGNOSTIC _sample] Before sampling: member_idx.shape={member_idx.shape}, non_member_idx.shape={non_member_idx.shape}")

    if max_samples_per_class is not None:
        if member_idx.shape[0] > max_samples_per_class:
            member_idx = rng.choice(member_idx, size=max_samples_per_class, replace=False)
        if non_member_idx.shape[0] > max_samples_per_class:
            non_member_idx = rng.choice(non_member_idx, size=max_samples_per_class, replace=False)

    sample_count = min(member_idx.shape[0], non_member_idx.shape[0])
    print(f"[MIA DIAGNOSTIC _sample] sample_count={sample_count}, max_samples_per_class={max_samples_per_class}")
    if sample_count < 10:
        print(f"[MIA DIAGNOSTIC _sample] WARNING: sample_count < 10, returning None")
        return None, None

    if member_idx.shape[0] > sample_count:
        member_idx = rng.choice(member_idx, size=sample_count, replace=False)
    if non_member_idx.shape[0] > sample_count:
        non_member_idx = rng.choice(non_member_idx, size=sample_count, replace=False)
    print(f"[MIA DIAGNOSTIC _sample] After sampling: member_idx.shape={member_idx.shape}, non_member_idx.shape={non_member_idx.shape}")
    return member_idx, non_member_idx


def _train_shadow_linear_model(X_train, y_train, train_mode, lam, num_steps, optimizer, lr, wd, random_state):
    torch.manual_seed(random_state)
    if train_mode == 'binary':
        b = torch.zeros(X_train.size(1), device=X_train.device)
        return lr_optimize(
            X_train, y_train, lam, b=b, num_steps=num_steps,
            verbose=False, opt_choice=optimizer, lr=lr, wd=wd
        )

    b = torch.zeros(X_train.size(1), y_train.size(1), device=X_train.device)
    return ovr_lr_optimize(
        X_train, y_train, lam, None, b=b, num_steps=num_steps,
        verbose=False, opt_choice=optimizer, lr=lr, wd=wd
    )


def membership_inference_attack(
    w, X, train_mask, test_mask, y=None, train_mode='ovr', lam=1e-3,
    shadow_num_steps=100, shadow_optimizer='Adam', shadow_lr=0.01, shadow_wd=5e-4,
    max_samples_per_class=20000, random_state=0
):
    """
    Posterior-shadow MIA adapted from rebMIGraph.
    Shadow train/out posteriors train the attack model, then target train/test posteriors are attacked.
    """
    # ========== 诊断日志：MIA 0.5 问题 ==========
    if y is None:
        print("[MIA DIAGNOSTIC] WARNING: y is None - returning 0.5")
        return 0.5

    member_idx, non_member_idx = _sample_member_nonmember_indices(
        train_mask, test_mask, max_samples_per_class=max_samples_per_class, random_state=random_state
    )
    if member_idx is None:
        print("[MIA DIAGNOSTIC] WARNING: member_idx or non_member_idx is None - returning 0.5")
        print(f"[MIA DIAGNOSTIC] train_mask sum = {train_mask.sum()}, test_mask sum = {test_mask.sum()}")
        return 0.5

    print(f"[MIA DIAGNOSTIC] member_idx shape = {member_idx.shape}, non_member_idx shape = {non_member_idx.shape}")
    
    rng = np.random.default_rng(random_state)
    target_member_idx = torch.from_numpy(member_idx).to(X.device)
    target_nonmember_idx = torch.from_numpy(non_member_idx).to(X.device)

    full_idx = np.arange(X.size(0))
    used_idx = np.concatenate([member_idx, non_member_idx])
    shadow_pool = np.setdiff1d(full_idx, used_idx, assume_unique=False)
    print(f"[MIA DIAGNOSTIC] X.shape = {X.shape}, shadow_pool.shape = {shadow_pool.shape}, required >= {2 * len(member_idx)}")
    if shadow_pool.shape[0] < 2 * len(member_idx):
        print(f"[MIA DIAGNOSTIC] WARNING: shadow_pool too small ({shadow_pool.shape[0]} < {2 * len(member_idx)}) - returning 0.5")
        return 0.5

    shadow_perm = rng.permutation(shadow_pool)
    shadow_member_idx = torch.from_numpy(shadow_perm[:len(member_idx)]).to(X.device)
    shadow_nonmember_idx = torch.from_numpy(shadow_perm[len(member_idx):len(member_idx) * 2]).to(X.device)

    X_shadow_train = X[shadow_member_idx]
    X_shadow_test = X[shadow_nonmember_idx]

    if train_mode == 'binary':
        y_all = y.to(X.device).float()
        y_shadow_train = y_all[shadow_member_idx]
    else:
        y_all = y.to(X.device).long()
        num_classes = int(y_all.max().item()) + 1
        y_shadow_train = (F.one_hot(y_all[shadow_member_idx], num_classes=num_classes) * 2 - 1).float()

    shadow_w = _train_shadow_linear_model(
        X_shadow_train, y_shadow_train, train_mode=train_mode, lam=lam,
        num_steps=shadow_num_steps, optimizer=shadow_optimizer, lr=shadow_lr,
        wd=shadow_wd, random_state=random_state
    )

    shadow_in_post = _posterior_from_linear_model(shadow_w, X_shadow_train).cpu().numpy()
    shadow_out_post = _posterior_from_linear_model(shadow_w, X_shadow_test).cpu().numpy()
    target_in_post = _posterior_from_linear_model(w, X[target_member_idx]).cpu().numpy()
    target_out_post = _posterior_from_linear_model(w, X[target_nonmember_idx]).cpu().numpy()

    X_attack = np.concatenate([shadow_in_post, shadow_out_post], axis=0)
    y_attack = np.concatenate([np.ones(shadow_in_post.shape[0]), np.zeros(shadow_out_post.shape[0])])
    X_target = np.concatenate([target_in_post, target_out_post], axis=0)
    y_target = np.concatenate([np.ones(target_in_post.shape[0]), np.zeros(target_out_post.shape[0])])

    attack_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state)
    attack_model.fit(X_attack, y_attack)

    mia_preds = attack_model.predict_proba(X_target)[:, 1]
    mia_auc = roc_auc_score(y_target, mia_preds)
    
    # ========== 深度诊断：理解 AUC < 0.5 的原因 ==========
    print(f"[MIA DEEP DIAG] Shadow model - in: mean={shadow_in_post.mean():.4f} ± {shadow_in_post.std():.4f}, "
          f"out: mean={shadow_out_post.mean():.4f} ± {shadow_out_post.std():.4f}")
    print(f"[MIA DEEP DIAG] Target model - in: mean={target_in_post.mean():.4f} ± {target_in_post.std():.4f}, "
          f"out: mean={target_out_post.mean():.4f} ± {target_out_post.std():.4f}")
    print(f"[MIA DEEP DIAG] Attack model coef={attack_model.coef_}, intercept={attack_model.intercept_}")
    print(f"[MIA DEEP DIAG] Attack preds - member: mean={mia_preds[:len(target_in_post)].mean():.4f}, "
          f"non-member: mean={mia_preds[len(target_in_post):].mean():.4f}")
    print(f"[MIA DEEP DIAG] Attack separation: |member_mean - nonmember_mean| = "
          f"{abs(mia_preds[:len(target_in_post)].mean() - mia_preds[len(target_in_post):].mean()):.4f}")
    
    return mia_auc
