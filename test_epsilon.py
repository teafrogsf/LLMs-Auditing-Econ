import math
import matplotlib.pyplot as plt
import numpy as np

def calculate_values():
    """
    计算不同epsilon值下的B、M、R值
    """
    # 设置参数
    T = 1000
    K = 3
    epsilon_values = [round(0.1 + i * 0.01, 2) for i in range(21)]  # 0.1到0.3，步长0.01
    
    # 存储结果
    results = {
        'epsilon': [],
        'B': [],
        'M': [],
        'R': []
    }
    
    print("计算不同epsilon值下的B、M、R值...")
    print("="*50)
    print(f"T = {T}, K = {K}")
    print()
    
    for epsilon in epsilon_values:
        # 计算各个变量
        B = T ** (2 * epsilon)
        M = (T ** (-epsilon)) * math.log(K * T)
        R = T - (2.3 + 3) * B * K  # (2.3+3) = 5.3
        
        # 存储结果
        results['epsilon'].append(epsilon)
        results['B'].append(B)
        results['M'].append(M)
        results['R'].append(R)
        
        print(f"ε = {epsilon:.2f}: B = {B:.6f}, M = {M:.6f}, R = {R:.6f}")
    
    print("\n计算完成！")
    return results

def plot_results(results):
    """
    绘制两个分离的折线图
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    epsilon_values = results['epsilon']
    B_values = results['B']
    M_values = results['M']
    R_values = results['R']
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 第一个图：B和M的折线图
    line1 = ax1.plot(epsilon_values, B_values, marker='^', linewidth=2, markersize=6, 
                     label='B', color='green', markerfacecolor='darkgreen')
    line2 = ax1.plot(epsilon_values, M_values, marker='o', linewidth=2, markersize=6, 
                     label='M', color='blue', markerfacecolor='darkblue')
    
    # 在数据点上添加数值标签 - B和M图
    for x, y in zip(epsilon_values, B_values):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0, 15), ha='center', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    for x, y in zip(epsilon_values, M_values):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0, -20), ha='center', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # 设置第一个图的属性
    ax1.set_xlabel('Epsilon (ε)', fontsize=12)
    ax1.set_ylabel('数值', fontsize=12)
    ax1.set_xticks(epsilon_values)
    ax1.set_xticklabels([f'{eps:.2f}' for eps in epsilon_values])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 第二个图：R的折线图
    line3 = ax2.plot(epsilon_values, R_values, marker='s', linewidth=2, markersize=6, 
                     label='R', color='red', markerfacecolor='darkred')
    
    # 在数据点上添加数值标签 - R图
    for x, y in zip(epsilon_values, R_values):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    # 设置第二个图的属性
    ax2.set_xlabel('Epsilon (ε)', fontsize=12)
    ax2.set_ylabel('数值', fontsize=12)
    ax2.set_xticks(epsilon_values)
    ax2.set_xticklabels([f'{eps:.2f}' for eps in epsilon_values])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('epsilon_analysis_separate_charts.png', dpi=300, bbox_inches='tight')
    print(f"\n图表已保存为: epsilon_analysis_separate_charts.png")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    print("LLMs-Auditing-Econ: Epsilon参数分析")
    print("="*50)
    print("新逻辑: T=1000, K=3, R=T-(2.3+3)BK")
    print("epsilon范围: 0.1到0.3，步长0.01")
    print()
    
    # 计算结果
    results = calculate_values()
    
    # 绘制图表
    plot_results(results)
    
    print("\n" + "="*50)
    print("分析完成！")
    print(f"epsilon范围: {min(results['epsilon']):.2f} - {max(results['epsilon']):.2f}")
    print(f"B值范围: {min(results['B']):.3f} - {max(results['B']):.3f}")
    print(f"M值范围: {min(results['M']):.3f} - {max(results['M']):.3f}")
    print(f"R值范围: {min(results['R']):.3f} - {max(results['R']):.3f}")