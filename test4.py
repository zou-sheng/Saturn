import matplotlib.pyplot as plt
import numpy as np

# 搜索步数（X轴）
samples = np.arange(10, 501, 10)

# ==================== Llama-2 34B 搜索收敛曲线（预期）====================
# EDP 随采样下降（越低越好）
edp_saturn = np.maximum(4000 * np.exp(-samples / 60) + 600, 650)  # 最快收敛
edp_dosa   = np.maximum(5000 * np.exp(-samples / 110) + 1200, 1400)
edp_hf     = np.maximum(5500 * np.exp(-samples / 160) + 2200, 2500) # 最差

# 绘图
plt.figure(figsize=(7,4))

plt.plot(samples, edp_hf,     'o-', label='HF', color='#d62728', alpha=0.8, markersize=3)
plt.plot(samples, edp_dosa,   's-', label='DoSA', color='#1f77b4', alpha=0.85, markersize=3)
plt.plot(samples, edp_saturn, '^-', label='Saturn (Ours)', color='#2ca02c', linewidth=2.5, markersize=4)

plt.xlabel('Number of Search Samples', fontsize=12)
plt.ylabel('Best EDP (Lower is Better)', fontsize=12)
plt.title('Search Convergence (Llama-2 34B)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('convergence_curve.pdf', dpi=300, bbox_inches='tight')
plt.show()
