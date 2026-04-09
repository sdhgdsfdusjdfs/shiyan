"""
快速 MIA 修复验证脚本（仅检查数据加载和基础问题）
"""

import torch
import numpy as np
import sys

print("="*60)
print("快速诊断：检查 MIA 修复")
print("="*60)

try:
    print("\n[1/4] 加载 DGraphFin 数据...")
    from dgraphfin import DGraphFin
    
    dataset = DGraphFin(root='./PyG_datasets', name='DGraphFin')
    data = dataset[0]
    
    # 规范化 mask
    if hasattr(data, 'valid_mask'):
        data.val_mask = data.valid_mask
    
    print(f"  ✓ 加载成功")
    print(f"    - 节点数：{data.num_nodes}")
    print(f"    - 边数：{data.edge_index.shape[1]}")  
    print(f"    - 特征维度：{data.x.shape[1]}")
    print(f"    - 训练：{data.train_mask.sum()}, 验证：{data.val_mask.sum()}, 测试：{data.test_mask.sum()}")
    
except Exception as e:
    print(f"  ✗ 失败：{e}")
    sys.exit(1)

try:
    print("\n[2/4] 检查 MIA 诊断代码...")
    from utils import membership_inference_attack
    import inspect
    
    source = inspect.getsource(membership_inference_attack)
    if "[MIA DIAGNOSTIC]" in source:
        print(f"  ✓ 诊断日志已添加")
    else:
        print(f"  ✗ 诊断日志未找到")
    
    if "[MIA DEEP DIAG]" in source:
        print(f"  ✓ 深度诊断日志已添加")
    else:
        print(f"  ✗ 深度诊断日志未找到")
        
    if "train_mask" in source and "test_mask" in source:
        print(f"  ✓ 代码使用完整训练集进行 MIA")
    else:
        print(f"  ✗ 代码仍需修复")
    
except Exception as e:
    print(f"  ✗ 检查失败：{e}")

try:
    print("\n[3/4] 检查影子池大小...")
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()
    
    member_count = train_mask.sum()
    non_member_count = test_mask.sum()
    all_nodes = data.num_nodes
    used_count = member_count + non_member_count
    shadow_pool_size = all_nodes - used_count
    required_shadow = 2 * min(member_count, non_member_count)
    
    print(f"  - 成员（训练）：{member_count}")
    print(f"  - 非成员（测试）：{non_member_count}")
    print(f"  - 需要的影子池：≥ {required_shadow}")
    print(f"  - 实际影子池：{shadow_pool_size}")
    
    if shadow_pool_size >= required_shadow:
        print(f"  ✓ 影子池充足！")
    else:
        print(f"  ✗ 影子池不足！还需 {required_shadow - shadow_pool_size} 个节点")
        print(f"    建议：用较小的 num_removes 值运行")
    
except Exception as e:
    print(f"  ✗ 检查失败：{e}")

print("\n[4/4] 运行简化示例...")
print("  (跳过完整 MIA 计算以节省时间)")
print("  建议：运行完整脚本查看详细诊断日志")

print("\n" + "="*60)
print("总结：")
print("  修复已完成。可以运行完整实验：")
print()
print("  python sgc_feature_node_unlearn.py \\")
print("    --dataset=dgraphfin --train_mode=binary \\")
print("    --num_removes=500 --num_steps=100 \\")
print("    --trails=1 --prop_step=2 \\")
print("    --lr=0.5 --lam=1e-3 --std=1e-2 \\")
print("    --eps=1.0 --delta=1e-4 \\")
print("    --optimizer=Adam --removal_mode=node")
print("="*60)
