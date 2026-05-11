import safetensors.torch
import torch
import numpy as np

def compare_safetensors(file1, file2, atol=1e-6, rtol=1e-6):
    """
    比较两个safetensors文件是否相同
    
    参数:
    - file1, file2: 文件路径
    - atol: 绝对误差容限
    - rtol: 相对误差容限
    """
    # 加载文件
    weights1 = safetensors.torch.load_file(file1)
    weights2 = safetensors.torch.load_file(file2)
    
    print(f"文件1的键数量: {len(weights1)}")
    print(f"文件2的键数量: {len(weights2)}")
    
    # 检查键的数量是否相同
    if len(weights1) != len(weights2):
        print(f"键数量不同: {len(weights1)} vs {len(weights2)}")
        return False
    
    # 检查所有键是否相同
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())
    
    if keys1 != keys2:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        if missing_in_2:
            print(f"文件2缺少键: {missing_in_2}")
        if missing_in_1:
            print(f"文件1缺少键: {missing_in_1}")
        return False
    
    # 比较每个张量
    all_close = True
    max_diff = 0
    max_diff_key = None
    
    for key in weights1:
        tensor1 = weights1[key]
        tensor2 = weights2[key]
        
        # 检查形状
        if tensor1.shape != tensor2.shape:
            print(f"键 '{key}' 的形状不同: {tensor1.shape} vs {tensor2.shape}")
            all_close = False
            continue
        
        # 计算差值
        diff = torch.abs(tensor1 - tensor2)
        current_max_diff = diff.max().item()
        
        if current_max_diff > max_diff:
            max_diff = current_max_diff
            max_diff_key = key
        
        # 使用allclose检查是否在容限范围内相等
        if not torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
            print(f"键 '{key}' 的值不同")
            print(f"  最大绝对差值: {current_max_diff:.6e}")
            print(f"  平均绝对差值: {diff.mean().item():.6e}")
            print(f"  形状: {tensor1.shape}")
            all_close = False
    
    print(f"\n最大差值: {max_diff:.6e} (键: '{max_diff_key}')")
    
    return all_close


# 使用示例
file1 = "/mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT/ema.safetensors"
file2 = "/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/rb0127overfit_0001000/ema_merged.safetensors"

result = compare_safetensors(file1, file2, atol=1e-6, rtol=1e-6)
print(f"\n两个文件是否相同: {result}")