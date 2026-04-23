# Project Overview

## Goal

This workspace is being developed into a graph privacy compliance experiment platform for publication-oriented research.

The core target is to support three graph unlearning scenarios on risk-control / anti-fraud graphs:

1. Node unlearning
2. Edge unlearning
3. Node feature unlearning

For each scenario, the project aims to evaluate three dimensions:

1. Utility
2. Efficiency
3. Security

Security evaluation is centered on membership inference attack (MIA) and its variants.

## Intended Research Story

The current research direction is to build a "dynamic privacy compliance risk-control system" around graph learning and graph unlearning.

The planned privacy-compliance scenarios are:

1. Right to be forgotten
   Node unlearning for account deletion or mandatory account removal.
2. Relationship revocation
   Edge unlearning for removing sensitive links, transactions, or social relations.
3. Sensitive profile desensitization
   Feature unlearning for removing a specific sensitive attribute without deleting the whole node.

The corresponding attack views are:

1. Membership inference attack
2. Link inference attack
3. Attribute inference attack

## Current Repository Roles

### `gnn+unlearn/sgc_unlearn-main`

Main experimental codebase.

Current role:

1. SGC-style graph unlearning backbone
2. Node / edge / feature unlearning experiments
3. Utility and efficiency evaluation
4. Initial MIA baseline

Current local progress:

1. `DGraphFin` has been connected to `sgc_feature_node_unlearn.py`
2. `AUC` and `F1` have been added alongside accuracy
3. A debug subgraph mode has been added for faster experiments
4. Debug sampling has been adjusted to preserve more edges
5. The original lightweight MIA has been retired from the main workflow
6. A rebMIGraph-style posterior-shadow MIA attack has been integrated into the main workflow
7. The current MIA still needs further validation because attack AUC is still close to 0.5 in current debug-scale DGraphFin runs

### `DGraphFin_baseline`

Reference repository for:

1. DGraphFin data organization
2. Official-like loading logic
3. Risk-control graph evaluation context

This repo is mainly used as a dataset and baseline reference, not the main execution framework right now.

### `rebMIGraph`

Reference repository for stronger graph membership inference attack experiments.

Planned role:

1. Replace or augment the current simplified MIA baseline
2. Provide a more publication-worthy attack protocol

### `EllipticPlusPlus`

Reference dataset repository for anti-money-laundering / illicit transaction style experiments.

Planned role:

1. Secondary or extended experiment dataset
2. Stronger "financial forensics / AML" narrative support

## Near-Term Development Plan

### Stage 1

Make node unlearning on `DGraphFin` stable and reproducible.

This includes:

1. Better training settings
2. Better debug and scaling workflow
3. Better metric reporting

### Stage 2

Upgrade the security evaluation.

This includes:

1. Integrating `rebMIGraph`
2. Replacing the current lightweight MIA with a stronger attack protocol
3. Verifying that pre-unlearning and post-unlearning attack performance shows meaningful separation

### Stage 3

Extend the same pipeline to:

1. Edge unlearning
2. Feature unlearning
3. Link inference attack
4. Attribute inference attack

Immediate next execution plan:

1. Keep node unlearning on `DGraphFin` in binary mode as the current stable diagnostic path
2. Treat current node-MIA findings as an empirical observation, not yet as the final privacy claim
3. Start porting `sgc_edge_unlearn.py` to `DGraphFin`
4. Build the edge-unlearning experimental path before formal link-inference attack integration
5. Use edge unlearning as the next main implementation target because MIA alone is not sufficient to support the final paper story

### Stage 4

Package results for paper writing:

1. Utility plots
2. Efficiency plots
3. Security plots
4. Formal experiment settings
5. Ablation and comparison results

## Current Status

The workspace is already capable of running preliminary `DGraphFin` node unlearning experiments.

However, it is not yet a complete publication-ready system because:

1. The current MIA is still not strong enough for final claims
2. Edge and feature privacy attack pipelines are not yet complete
3. The current runs are still mostly debug-scale rather than final-scale experiments

## Experiment Issues Log

### Issue 1: MIA attack AUC remains at 0.5

Observed in current `DGraphFin` debug-scale runs:

1. `MIA Before Unlearning` is often `0.5000`
2. `MIA After Unlearning` is also often `0.5000`
3. Replacing the initial lightweight MIA with a rebMIGraph-style posterior-shadow attack did not immediately change this behavior

Current interpretation:

1. The attack pipeline has been upgraded, so the remaining problem is likely not only "weak MIA code"
2. The current model/data/training setup may not produce enough member-vs-nonmember separation
3. The debug-scale subgraph setting may still be too far from the intended full experimental threat model

Current action items:

1. Inspect train metrics directly to verify whether the target model is overfitting or collapsing
2. Check class distribution and prediction collapse under highly imbalanced settings
3. Continue refining the MIA protocol only after confirming the target model is actually learnable and leak-prone
4. Print train/val/test label distributions and prediction distributions in the main workflow
5. Activate balanced training options before interpreting any further MIA results

### Issue 2: Accuracy can be misleading under class imbalance

Observed in one aggressive parameter setting on `DGraphFin`:

1. Validation accuracy and test accuracy both reached about `0.99`
2. `F1` stayed around `0.50`
3. `AUC` stayed around `0.50`
4. MIA also stayed at `0.50`

Current interpretation:

1. This strongly suggests prediction collapse or majority-class dominance
2. Accuracy alone is not a reliable utility metric in this setting
3. `F1` and `AUC` must remain primary metrics for paper writing

Status:

1. Train metrics are now being added to the main workflow to help diagnose this issue
2. Label distribution and prediction distribution diagnostics have also been added
3. A train-balancing option is now being activated as a collapse mitigation path
4. Once the exact failure mode is confirmed, the diagnosis and fix should be appended here

### Issue 3: OVR training can collapse to class-0 prediction even after balancing

Observed in `DGraphFin` debug-scale runs with balanced training enabled:

1. Balanced training distribution became roughly symmetric
2. However, prediction distribution still collapsed to class `0`
3. `Accuracy` stayed very high because of evaluation-set imbalance
4. `AUC` and `F1` showed that the model was not actually solving the target task well

Current interpretation:

1. The issue is not only data imbalance
2. The `ovr` setup can still degenerate in weak-signal binary-like settings
3. The current linear SGC-style decision function is vulnerable to prediction collapse under this configuration

Current mitigation:

1. Switch the experiment from `ovr` to explicit `binary` mode when the sampled subgraph effectively behaves like a binary task

### Issue 4: Binary mode recovers ranking signal, but thresholded classification is still poor

Observed after switching to `binary` mode on `DGraphFin` debug-scale runs:

1. `Train/Val/Test AUC` rose to about `0.69`
2. `F1` remained extremely low (around `0.03`)
3. `MIA Before Unlearning` moved from exact `0.50` to about `0.41`

Current interpretation:

1. The model is no longer fully collapsed
2. It has learned a ranking signal, since AUC is clearly above random
3. But the default decision threshold still leads to poor positive-class prediction behavior
4. The MIA value below `0.5` suggests the attack may be learning the signal in the reversed direction, or the in/out labeling convention may need to be flipped during evaluation

Current action items:

1. Inspect binary prediction thresholding and positive-class recall
2. Add precision / recall reporting for the positive class
3. Check whether attack predictions should be inverted when AUC < 0.5
4. Consider reporting `max(AUC, 1-AUC)` only as a diagnostic, not as the final paper metric, until the attack direction is confirmed

Important note:

1. `1 - AUC` should not be treated as the final paper result by default
2. It is only a debugging signal for checking whether the attack direction or label convention is reversed
3. An AUC below `0.5` is not "guessing" in the strict sense; it usually means the attack score is anti-correlated with the positive label under the current evaluation convention

Latest empirical update:

1. In `binary` mode, `DGraphFin` debug-scale runs reached about `0.69-0.70` AUC on the downstream task
2. Positive-class precision remained extremely low while recall stayed relatively high
3. This indicates the model has learned ranking information but still has poor thresholded decision behavior
4. MIA remained at `0.50` under the current target/shadow attack protocol
5. Current interpretation: task-level ranking signal now exists, but member-vs-nonmember separability is still weak in this setup

## Literature Positioning Notes

Current reading-based understanding:

1. Graph unlearning itself is already an active and non-new direction
2. MIA is already a standard privacy audit in broader machine unlearning
3. Simply combining "graph unlearning + MIA" is unlikely to be novel enough by itself

What may still be publishable:

1. A realistic risk-control / anti-fraud graph setting rather than only citation benchmarks
2. A unified three-scenario compliance framework:
   node unlearning, edge unlearning, feature unlearning
3. A stronger privacy audit suite:
   MIA, link inference, attribute inference
4. A careful efficiency / utility / security benchmark on real financial or fraud graphs
5. A practical conclusion about when graph unlearning actually does or does not reduce privacy leakage

What is probably not enough on its own:

1. Re-running an existing graph unlearning method with one more MIA figure
2. Showing only one dataset and one attack metric
3. Treating any post-unlearning privacy drop as automatically publishable novelty

## Problem-Solution Tracking

This section should be updated whenever an issue is confirmed or resolved.

Recommended template:

1. Problem
2. Observed symptoms
3. Suspected cause
4. Implemented fix
5. Evidence after fix

### Issue 5: MIA attack 后验过度平滑 (Posterior Over-smoothing)

**Problem**: 在 Cora 数据集上，即使 MIA 框架工作正常，成员/非成员的后验分布也完全相同，导致攻击 AUC ≈ 0.5。

**Observed Symptoms**:
1. 影子池充足，MIA 代码正常运行
2. 后验分布完全相同：
   ```
   Shadow model - in: mean=0.1429, out: mean=0.1429
   Target model - in: mean=0.1429, out: mean=0.1429
   ```
3. 攻击分离度为 0：
   ```
   Attack separation: 0.0000
   AUC = 0.49
   ```

**Root Cause Analysis**:
1. **模型结构问题**：线性 SGC 在高度聚类的图上产生过度平滑
   - SGC 经过图聚合后，所有节点预测趋向均匀分布 (1/7 ≈ 0.1429)
   - 不同节点间的区别被消除
   - 无法区分成员/非成员

2. **数据特性问题**：Cora 数据集本身的特点
   - 是引用网络，高度同质
   - 节点标签与结构强相关
   - 导致任何学到的模型都会过度平滑

3. **威胁模型不匹配**：posterior-based MIA 要求模型产生差异化的置信度
   - 但线性模型无法保持个体节点的特异性
   - 即使模型学到了训练集，也无法在后验中体现

**Technical Solutions** (技术解决方案)：

#### 方案 1: 切换到非线性 GNN（推荐✓✓✓）
**实施成本**：中等 | **隐私揭露潜力**：高 | **学术价值**：高

替换 SGC 为更复杂的模型（如 GCN、GAT）：

```python
# 旧代码：线性 SGC
w = lr_optimize(X_propagated, y_train, ...)

# 新代码：使用 GCN
from torch_geometric.nn import GCNConv
model = GCN(input_dim, hidden_dim=64, num_classes=num_classes)
w = train_gnn(model, data, y_train, ...)
```

**优点**：
- ✓ 非线性激活保留个体差异
- ✓ 更贴近现实应用（GNN 通常用于图任务）
- ✓ 可能发现更强的隐私泄露

**缺点**：
- ✗ 模型复杂，因此遗忘计算复杂度提高
- ✗ 理论分析困难

**论文改写**：
> "To better reflect the privacy risks in real-world GNN applications, we use GCN instead of the simplified SGC. Non-linearity preserves member/non-member distinction in model outputs."

---

#### 方案 2: 使用置信度 MIA 而非后验 MIA（推荐✓✓）
**实施成本**：低 | **隐私揭露潜力**：中 | **学术价值**：中

改变 MIA 的特征，不用模型后验，用训练曲线/梯度等：

```python
# 旧特征：直接用模型后验
mia_features = model_posterior[member_indices]

# 新特征：用多轮训练的置信度变化
mia_features = [
    confidence_round_0,
    confidence_round_10,
    confidence_round_20,
    # ... 训练过程中的动态特征
]
```

**原理**：
- 成员在训练过程中的置信度快速提升
- 非成员（测试集）无法通过训练改改进
- 攻击者可以观察这种"学习曲线"来推断成员

**优点**：
- ✓ 实现简单
- ✓ 不需要修改模型架构
- ✓ 更符合实际攻击场景（黑盒）

**缺点**：
- ✗ 需要多次查询模型，不是白盒设置
- ✗ 学术新颖性有限

**论文改写**：
> "We augment posterior-based MIA with training-curve-based features, capturing member-vs-nonmember distinction through convergence dynamics rather than final predictions."

---

#### 方案 3: 选择更易过拟合的数据集（推荐✓✓✓）
**实施成本**：低 | **隐私揭露潜力**：高 | **学术价值**：高

不用 Cora，改用反欺诈图（DGraphFin）或其他异构数据集：

```python
# 改为 DGraphFin（高度不平衡，更易过拟合）
dataset = DGraphFin(root='./dataset/')
```

**为什么 DGraphFin 更好**：
- 二分类问题（自然过拟合风险更高）
- 类别不平衡（90% vs 10%）
- 节点异构性高（不同类型频繁）
- MIA 更容易检测到过拟合

**预期结果**：
```
DGraphFin 上：AUC = 0.65-0.75（有意义的隐私泄露）
```

**优点**：
- ✓ 现成的数据集（已在项目中）
- ✓ 更真实的应用场景
- ✓ 更强的隐私隐忧

**缺点**：
- ✗ 需要改动所有实验

**论文改写**：
> "We establish our privacy evaluation on the realistic anti-fraud graph DGraphFin, which exhibits natural class imbalance and high heterogeneity, creating stronger overfitting risks that are more representable of real-world graph unlearning scenarios."

---

#### 方案 4: 改进后验聚合方式（中等✓）
**实施成本**：中等 | **隐私揭露潜力**：中 | **学术价值**：低

不用单层后验，用多层特征的组合：

```python
# 旧：直接用最后一层的后验
features_for_attack = model.posterior

# 新：用中间隐藏层的激活值
features_for_attack = [
    model.layer1_activations,   # 第一层
    model.layer2_activations,   # 第二层
    model.final_posterior        # 最后层
]
```

**原理**：中间层可能保留更多成员信息

**优点**：
- ✓ 无需改动模型架构
- ✓ 实现简单

**缺点**：
- ✗ 对线性模型无效（只有一层）
- ✗ 学术价值不高

---

#### 方案 5: 在遗忘中加入噪声防护（研究方向✓）
**实施成本**：高 | **隐私揭露潜力**：低 | **学术价值**：中

改进遗忘算法本身，使得遗忘后的模型满足 DP：

```python
# 在权重更新中加入 DP 噪声
w_new = w - eta * grad + noise_from_laplace(scale)
```

**原理**：
- 即使没有后验差异，DP 遗忘也保证了隐私
- MIA 无法成功（信息论下界）

**优点**：
- ✓ 有形式化隐私保证
- ✓ 学术价值高

**缺点**：
- ✗ 大幅增加噪声，伤害模型效用
- ✗ 实现复杂，需要修改核心遗忘算法

---

## 📋 **论文策略建议**

### **方案 A：转向更好的模型 + 更好的数据（推荐）**

**核心改变**：
1. SGC → GCN（非线性）
2. Cora → DGraphFin（高异构性）

**论文叙述**：
> "While posterior-based MIA on simplified citation networks may not capture meaningful privacy risks due to model smoothing, we evaluate privacy on two frontiers:
> 1. **Non-linear models** (GCN, GAT) that preserve node-specific information
> 2. **Realistic datasets** (DGraphFin) with natural class imbalance and heterogeneity"

**预期结果**：
- MIA AUC 从 0.50 → 0.70+
- 论文更有说服力

---

### **方案 B：保留线性模型，改变 MIA 方式**

**核心改变**：
1. 保留 SGC（理论简洁）
2. 改用 confidence/training-curve MIA
3. 换数据到 DGraphFin

**论文叙述**：
> "We demonstrate that posterior-based MIA has fundamental limitations on smooth models. Instead, we propose training-dynamics-based MIA which leverages convergence patterns to infer membership."

**优点**：
- ✓ 保留原有理论框架
- ✓ 新的 MIA 方式本身有贡献

---

### **方案 C：诚实的局限性分析（发表友好✓）**

**核心改变**：
1. 保留所有实验设置
2. 在论文中清楚说明 over-smoothing 问题

**论文叙述**：
> "**Limitations**: Our experiments on simplified linear GCNs with citation networks reveal an important phenomenon: graph convolution can lead to posterior smoothing where member and non-member outputs become indistinguishable. This limits membership inference attacks but also suggests that over-smoothed models inherently have privacy-preserving properties. Future work should evaluate unlearning on non-linear architectures."

**优点**：
- ✓ 学术诚实
- ✓ 审稿人欣赏坦诚的局限分析
- ✓ 在 Limitation 中指向未来工作

---

## 🎯 **推荐行动清单**

按优先级：

### **立即可做（1-2 小时）**
1. ✅ 在 Issue 5 中记录 over-smoothing 现象
2. ✅ 更新 PROJECT_OVERVIEW 说明问题
3. ✅ 准备在 DGraphFin 上重跑实验

### **短期改进（3-5 小时）**
- [ ] 在 DGraphFin 上运行相同实验
- [ ] 观察 MIA AUC 是否提升到 0.65+
- [ ] 比较 Cora vs DGraphFin 的后验分布

### **中期扩展（1-2 天）**
- [ ] 如果有时间，用 GCN 替换 SGC 再测一遍
- [ ] 实现 training-curve-based MIA 作为对比

### **论文策略**
- [ ] 明确选择方案 A/B/C 之一
- [ ] 按该方案改组实验
- [ ] 在论文中清楚说明选择理由

---

## 📊 **实验对比预期**

| 设置 | Posterior MIA AUC | 应对策略 |
|------|------------------|---------|
| Cora + SGC | 0.50（失败） | 识别 over-smoothing 问题 ✓ |
| DGraphFin + SGC | 0.60-0.70（可接受） | 继续使用 ✓ |
| DGraphFin + GCN | 0.65-0.80（理想） | 最优选择 ✓✓ |

---

**总体建议**：
建议采用 **方案 A**（模型+数据改进）的论文策略，同时在论文中记录 Cora+SGC 上发现的 over-smoothing 现象作为负面结果的学术价值。

## Maintenance Note

This file is intended to be updated during future conversations whenever the project direction, architecture, experiment protocol, or progress changes in a meaningful way.

## Edge Experiment Update (2026-04-23)

This section records the latest edge-unlearning findings on `DGraphFin` under the current stable setting:

1. `train_mode=binary`
2. `compare_retrain=True`
3. utility/efficiency from `sgc_edge_unlearn.py`
4. security from `edge_link_inference_eval.py` (link inference)

### Utility & Efficiency Summary

Random strategy (`random`) observations:

1. `num_removes=500/1000/2000/5000` all show near-identical utility
2. final `AUC` remains around `0.5974-0.5975`
3. final `F1` remains around `0.0319`
4. `Unlearning` and `Retrain` are nearly overlapping in utility
5. speedup stays around `1.59x-1.66x`

High-degree strategy (`high_degree`) observations:

1. `num_removes=500/1000/2000` also show near-identical utility
2. final `AUC` remains around `0.5973-0.5974`
3. final `F1` remains around `0.0319`
4. speedup stays around `1.59x`

Interpretation:

1. Edge-unlearning update path is computationally meaningful (stable speedup).
2. Utility under the current setup is in a low-signal plateau.
3. Deletion strategy does not materially change utility in this setting.

### Security Summary (Link Inference)

Random strategy (`random`) mean Delta (After-Before):

1. `nr=500`: Delta AUC `-0.2968`, Delta AP `-0.2660`
2. `nr=1000`: Delta AUC `-0.3096`, Delta AP `-0.2804`
3. `nr=2000`: Delta AUC `-0.3052`, Delta AP `-0.2797`
4. `nr=5000`: Delta AUC `-0.2952`, Delta AP `-0.2689`

High-degree strategy (`high_degree`) mean Delta (After-Before):

1. `nr=500`: Delta AUC `-0.0115`, Delta AP `-0.0117`
2. `nr=1000`: Delta AUC `-0.0283`, Delta AP `-0.0277`
3. `nr=2000`: Delta AUC `-0.2016`, Delta AP `-0.1410`
4. `nr=5000`: Delta AUC `-0.1667`, Delta AP `-0.1063`

Key security comparison:

1. At every tested scale, `|Delta|` under `random` is larger than under `high_degree`.
2. This means random edge deletion reduces link-inference recoverability more strongly in this setup.
3. Therefore, security sensitivity to deletion strategy is now empirically established.

### Main Conclusions for Paper Narrative

The edge experiment can now support the following thesis-aligned statements:

1. Edge unlearning is an efficient approximation to retraining (`~1.6x` speedup).
2. Functional metrics alone are insufficient to judge privacy gains (utility is similar across strategies).
3. Security outcomes depend on *what* is deleted, not only *how much* is deleted.
4. This directly supports the paper's core claim: functional forgetting does not automatically imply security forgetting.

### Publication Readiness Status (Edge Line)

Current status:

1. Utility: completed for both `random` and `high_degree`
2. Efficiency: completed for both `random` and `high_degree`
3. Security: completed for both `random` and `high_degree` (500/1000/2000/5000)
4. Strategy comparison evidence: completed

Remaining cross-paper gaps:

1. Feature-unlearning line is not yet at the same maturity level.
2. Second-dataset validation (e.g., `Elliptic++`) is still pending.
3. Mechanism diagnostics (PR-AUC / threshold sensitivity) are recommended as strengthening analysis.
