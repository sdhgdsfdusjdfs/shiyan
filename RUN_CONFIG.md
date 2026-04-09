# 推荐的 MIA 调试参数配置

## 问题分析
当前 MIA AUC = 0.5 是因为：
1. 样本不足：只用被删除的 50 个节点做成员集太小
2. 影子集不足：导致直接返回 0.5

## 解决方案
使用**完整训练集进行 MIA**（而不是只用删除的节点），同时用**二进制模式**确保更好的收敛。

---

## 配置 A：快速测试（运行时间 5-15 分钟）

用于快速验证 MIA 修复是否有效：

```bash
python sgc_feature_node_unlearn.py \
  --dataset='dgraphfin' \
  --train_mode='binary' \
  --num_removes=100 \
  --num_steps=50 \
  --trails=1 \
  --prop_step=2 \
  --lr=0.5 \
  --lam=1e-3 \
  --std=1e-2 \
  --eps=1.0 \
  --delta=1e-4 \
  --disp=10 \
  --removal_mode='node' \
  --optimizer='Adam' \
  --data_dir='./PyG_datasets'
```

**特点：**
- num_removes=100：足够大（100 个成员 + 100+ 非成员 = ~200 样本）
- num_steps=50：较低迭代数，快速收敛
- trails=1：仅 1 次重复，节省时间
- binary 模式：更稳定的收敛

---

## 配置 B：标准实验（运行时间 30-60 分钟）

用于获得可靠的 MIA 性能数据：

```bash
python sgc_feature_node_unlearn.py \
  --dataset='dgraphfin' \
  --train_mode='binary' \
  --num_removes=500 \
  --num_steps=100 \
  --trails=3 \
  --prop_step=2 \
  --lr=0.5 \
  --lam=1e-3 \
  --std=1e-2 \
  --eps=1.0 \
  --delta=1e-4 \
  --disp=50 \
  --removal_mode='node' \
  --optimizer='Adam' \
  --data_dir='./PyG_datasets'
```

**特点：**
- num_removes=500：较大规模（500 个成员）
- num_steps=100：标准迭代数
- trails=3：3 次重复，获得可靠统计
- binary 模式稳定

---

## 配置 C：完整实验（运行时间 1-2 小时）

用于最终发表的论文实验：

```bash
python sgc_feature_node_unlearn.py \
  --dataset='dgraphfin' \
  --train_mode='binary' \
  --num_removes=1000 \
  --num_steps=200 \
  --trails=5 \
  --prop_step=2 \
  --lr=0.5 \
  --lam=1e-3 \
  --std=1e-2 \
  --eps=1.0 \
  --delta=1e-4 \
  --disp=100 \
  --removal_mode='node' \
  --optimizer='Adam' \
  --data_dir='./PyG_datasets' \
  --compare_retrain
```

**特点：**
- num_removes=1000：较大规模
- num_steps=200：更多迭代，更好收敛
- trails=5：5 次重复，统计置信度高
- 启用完全重训练比较

---

## 关键参数说明

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `--dataset` | `dgraphfin` | 大型反欺诈图，~5000+ 节点，适合 MIA |
| `--train_mode` | `binary` | 二进制模式比 ovr 更稳定（见 Issue 4） |
| `--num_removes` | 100-1000 | 足以支撑 MIA：member≈num_removes，non-member≈test_mask |
| `--num_steps` | 50-200 | shadow 模型和主模型的训练步数 |
| `--trails` | 1-5 | 重复次数；快速测试用 1，论文用 3-5 |
| `--lam` | 1e-3 | L2 正则化系数 |
| `--std` | 1e-2 | 初始权重标准差 |
| `--prop_step` | 2 | 图传播步数；2-3 足够 |
| `--optimizer` | `Adam` | 比 LBFGS 更快 |
| `--lr` | 0.5 | Adam 学习率 |

---

## 预期结果

修复后的 MIA（使用完整训练集）应该：

- ✅ **MIA Before Unlearning**: AUC > 0.7（而不是 0.5）
- ✅ **MIA After Unlearning**: AUC 降低（显示隐私改进）
- ✅ **二进制模式任务 AUC**: ~0.69-0.75
- ✅ **运行时间**: 配置 A 约 10 分钟内完成

---

## 运行步骤

### 1. 快速验证（推荐从这里开始）
```bash
cd d:\论文代码\gnn+unlearn\sgc_unlearn-main
python sgc_feature_node_unlearn.py --dataset='dgraphfin' --train_mode='binary' --num_removes=100 --num_steps=50 --trails=1 --lam=1e-3 --std=1e-2 --eps=1.0 --delta=1e-4 --optimizer='Adam' --data_dir='./PyG_datasets'
```

### 2. 查看输出
运行时应该看到：
- `[MIA DIAGNOSTIC _sample] ...` 诊断日志
- `[MIA DIAGNOSTIC] member_idx shape = (100, ), non_member_idx shape = (...)` 
- `[MIA Before Unlearning] attack AUC = X.XXXX`（应该 > 0.5）
- `[MIA After Unlearning] attack AUC = Y.YYYY`（应该 < X.XXXX）

### 3. 如果仍然 = 0.5
说明需要进一步修复代码中的 MIA 实现
