import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

def parse_mse_loss(filename):
    """
    解析日志文件，提取步数和MSE Loss
    """
    steps = []
    mse_losses = []
    
    # 正则表达式匹配MSE Loss
    pattern = re.compile(r'step=(\d+)\).*?Train Loss mse: ([\d.]+)')
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                mse = float(match.group(2))
                steps.append(step)
                mse_losses.append(mse)
    
    return steps, mse_losses

def moving_average(data, window_size=3):
    """
    计算移动平均
    """
    if window_size < 2:
        return data
    
    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    smoothed = []
    
    for i in range(len(data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)
        window_data = data[start_idx:end_idx]
        smoothed.append(np.mean(window_data))
    
    return smoothed

def plot_smooth_mse_loss(steps, mse_losses, save_path=None, smooth_method='moving_average', smooth_window=3):
    """
    绘制平滑后的MSE Loss曲线
    """
    plt.figure(figsize=(14, 7))
    
    # 应用平滑
    if smooth_method == 'moving_average':
        smoothed_losses = moving_average(mse_losses, smooth_window)
        smooth_label = f'Smoothed (MA, window={smooth_window})'
    elif smooth_method == 'gaussian':
        # 高斯平滑，sigma值控制平滑程度
        sigma = smooth_window / 3  # 根据窗口大小计算sigma
        smoothed_losses = gaussian_filter1d(mse_losses, sigma=sigma)
        smooth_label = f'Smoothed (Gaussian, σ={sigma:.1f})'
    else:
        smoothed_losses = mse_losses
        smooth_label = 'Original'
    
    # 绘制原始曲线（半透明）
    plt.plot(steps, mse_losses, 'gray', linewidth=1, alpha=0.3, label='Original', marker='o', markersize=3)
    
    # 绘制平滑后的曲线（更粗更明显）
    plt.plot(steps, smoothed_losses, 'b-', linewidth=2.5, label=smooth_label)
    
    # 高亮显示最小Loss点
    min_idx = np.argmin(mse_losses)
    plt.scatter(steps[min_idx], mse_losses[min_idx], color='red', s=100, 
                zorder=5, label=f'Min Loss: {mse_losses[min_idx]:.4f}')
    
    # 高亮显示最大Loss点
    max_idx = np.argmax(mse_losses)
    plt.scatter(steps[max_idx], mse_losses[max_idx], color='orange', s=100, 
                zorder=5, label=f'Max Loss: {mse_losses[max_idx]:.4f}')
    
    # 设置图表属性
    plt.title('Train Loss (MSE) - Original vs Smoothed', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    
    # 设置x轴为整数刻度
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # 添加统计信息
    min_loss = min(mse_losses)
    max_loss = max(mse_losses)
    avg_loss = np.mean(mse_losses)
    final_loss = mse_losses[-1]
    final_smoothed = smoothed_losses[-1]
    
    stats_text = (f'Original:\n'
                  f'Min: {min_loss:.4f}\n'
                  f'Max: {max_loss:.4f}\n'
                  f'Avg: {avg_loss:.4f}\n'
                  f'Final: {final_loss:.4f}\n\n'
                  f'Smoothed:\n'
                  f'Final: {final_smoothed:.4f}')
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()
    
    # 打印统计数据
    print("\n" + "="*60)
    print("MSE Loss 统计数据和趋势分析:")
    print("="*60)
    print(f"总步数: {len(steps)}")
    print(f"原始Loss范围: {min_loss:.4f} - {max_loss:.4f} (跨度: {max_loss-min_loss:.4f})")
    print(f"原始平均Loss: {avg_loss:.4f}")
    print(f"原始最终Loss: {final_loss:.4f}")
    print(f"平滑最终Loss: {final_smoothed:.4f}")
    
    # 趋势分析
    if len(mse_losses) >= 10:
        # 查看最后几个点的趋势
        last_5 = mse_losses[-5:]
        last_5_trend = np.polyfit(range(5), last_5, 1)[0]
        
        print(f"\n趋势分析:")
        print(f"最近5步Loss: {[f'{x:.4f}' for x in last_5]}")
        print(f"最近5步趋势斜率: {last_5_trend:.6f}")
        if last_5_trend < -0.001:
            print("趋势: 📉 下降 (训练有效)")
        elif last_5_trend > 0.001:
            print("趋势: 📈 上升 (可能需要调整)")
        else:
            print("趋势: ↔️ 稳定")

# 只绘制平滑曲线的简洁版本
def plot_simple_smooth_mse_loss(steps, mse_losses, smooth_window=5):
    """
    只绘制平滑后的MSE Loss曲线（简洁版本）
    """
    plt.figure(figsize=(12, 6))
    
    # 计算移动平均
    smoothed_losses = moving_average(mse_losses, smooth_window)
    
    # 绘制平滑后的曲线
    plt.plot(steps, smoothed_losses, 'b-', linewidth=2.5, label=f'Smoothed MSE Loss (window={smooth_window})')
    
    # 设置图表属性
    plt.title(f'Smooth Train Loss (MSE) - Moving Average (window={smooth_window})', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    
    # 设置x轴为整数刻度
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # 添加统计信息
    min_loss = min(smoothed_losses)
    max_loss = max(smoothed_losses)
    avg_loss = np.mean(smoothed_losses)
    final_loss = smoothed_losses[-1]
    
    stats_text = f'Min: {min_loss:.4f}\nMax: {max_loss:.4f}\nAvg: {avg_loss:.4f}\nFinal: {final_loss:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('ex.png')

# 使用示例
if __name__ == "__main__":
    # 请将 'your_log_file.txt' 替换为你的实际文件名
    filename = '/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/rb0130_indoor/log.txt'  # 或者使用完整的文件路径

    # 解析数据
    steps, mse_losses = parse_mse_loss(filename)
    

    plot_simple_smooth_mse_loss(steps, mse_losses, smooth_window=1)
