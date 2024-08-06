import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体属性
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 20  # 设置标题字体大小
plt.rcParams['axes.labelsize'] = 18  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 14  # 设置x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 14  # 设置y轴刻度字体大小

# 定义数据
models = ['Transformer', 'LSTM', 'MLP', 'DyGSP-mamba']
ap_scores = [[0.9578, 0.9580, 0.9591, 0.9581]]

# 绘制AP得分箱型图
fig, ax = plt.subplots(figsize=(10, 6))
bplot = ax.boxplot(ap_scores, patch_artist=True, widths=0.5)

# 设置箱型图颜色
colors = ['gray', 'gray', 'orange', 'gray']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# 设置x轴标签
ax.set_xticklabels(models)
ax.set_title('Average Precision (AP) Scores')
ax.set_ylabel('AP Score')
ax.set_ylim(0.955, 0.960)

plt.tight_layout()
plt.savefig('AP_Scores_Boxplot.png')
plt.show()
