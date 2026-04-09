"""
独立 MIA 诊断脚本
直接测试修复后的 MIA 实现
"""

import torch
import numpy as np
from dgraphfin import DGraphFin
from utils import membership_inference_attack

# 加载数据集
print("="*50)
print("加载 DGraphFin 数据")
print("="*50)
dataset = DGraphFin(root='./PyG_datasets', name='DGraphFin')
data = dataset[0]

# 规范化 mask
if hasattr(data, 'valid_mask'):
    data.val_mask = data.valid_mask

print(f"数据集大小：{data.x.shape}")
print(f"节点数：{data.num_nodes}")
print(f"边数：{data.edge_index.shape[1]}")
print(f"特征维度：{data.x.shape[1]}")
print(f"训练集：{data.train_mask.sum()}")
print(f"验证集：{data.val_mask.sum()}")
print(f"测试集：{data.test_mask.sum()}")
print(f"类别数：{int(data.y.max().item()) + 1}")

print("\n" + "="*50)
print("测试 MIA 攻击（使用完整训练集）")
print("="*50)

# 生成一个简单的线性分类器作为目标模型
from sklearn.linear_model import LogisticRegression

X = data.x.numpy()
y = data.y.numpy()
train_mask = data.train_mask.numpy()
test_mask = data.test_mask.numpy()
val_mask = data.val_mask.numpy()

# 训练目标模型
X_train = X[train_mask]
y_train = y[train_mask]

print(f"\n训练目标模型...")
model = LogisticRegression(max_iter=1000, multi_class='ovr' )
model.fit(X_train, y_train)

# 转换为 torch 张量（MIA 期望的格式）
X_torch = torch.from_numpy(X).float()
y_torch = torch.from_numpy(y).long()
w = torch.from_numpy(model.coef_).float()  # 权重矩阵

print(f"训练完成。模型准确率（训练集）：{model.score(X_train, y_train):.4f}")

# 运行 MIA
print(f"\n运行 MIA 攻击...")
print(f"  - 成员集：train_mask ({train_mask.sum()} 个节点)")
print(f"  - 非成员集：test_mask ({test_mask.sum()} 个节点)")

mia_auc = membership_inference_attack(
    w, X_torch, train_mask, test_mask, y=y_torch, 
    train_mode='ovr', lam=1e-3, 
    shadow_num_steps=100, shadow_optimizer='Adam',
    shadow_lr=0.01, shadow_wd=5e-4
)

print(f"\n{'='*50}")
print(f"MIA 结果：AUC = {mia_auc:.4f}")
print(f"{'='*50}")

if mia_auc > 0.5:
    print(f"✓ 攻击成功！AUC > 0.5 表示隐私泄露")
elif mia_auc < 0.5:
    print(f"⚠ 攻击方向可能反向（AUC < 0.5）")
    print(f"   通常表示模型学到了反向的因果关系")
    print(f"   1-AUC = {1-mia_auc:.4f}")
else:
    print(f"! 完全随机（AUC = 0.5）")

print(f"\n如果 AUC ≠ 0.5，说明修复成功！")
print(f"如果仍为 0.5，检查上面的诊断日志 [MIA DIAGNOSTIC] 和 [MIA DEEP DIAG]")
