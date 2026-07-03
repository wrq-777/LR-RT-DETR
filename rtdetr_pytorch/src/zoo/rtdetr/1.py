import matplotlib.pyplot as plt
import numpy as np

# 1. GC10-DET 数据准备 (真实统计数据)
# 这是一个典型的长尾分布数据集
categories = [
    'Punching (Pu)',       # 冲孔 (949) - 最多
    'Silk Spot (Ss)',      # 丝斑 (663)
    'Inclusion (In)',      # 夹杂 (469)
    'Weld Line (Wl)',      # 焊缝 (258)
    'Crescent Gap (Cg)',   # 月牙湾 (229)
    'Oil Spot (Os)',       # 油斑 (215)
    'Water Spot (Ws)',     # 水斑 (203)
    'Waist Folding (Wf)',  # 腰折 (159)
    'Crease (Cr)',         # 折痕 (146)
    'Rolled Pit (Rp)'      # 压痕 (39) - 极少！
]

# 对应的真实数量
counts = [949, 663, 469, 258, 229, 215, 203, 159, 146, 39]

# 2. 创建画布
fig, ax = plt.subplots(figsize=(12, 6)) # 稍微宽一点，因为有10个类

# 3. 绘制柱状图
# 使用相同的学术蓝
bar_color = '#5b9bd5'
bars = ax.bar(categories, counts, color=bar_color, width=0.6, edgecolor='none')

# 4. 特殊处理：高亮最少的类别 (Rp)，强调长尾问题 (可选，不需要可删除这行)
bars[-1].set_color('#ed7d31') # 把最少的 Rolled Pit 标成橙色，突出难点

# 5. 在每个柱子上方添加数量标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 10,
            f'{height}',
            ha='center', va='bottom', fontsize=11, color='black', fontweight='bold')

# 6. 设置标题和坐标轴
ax.set_title('GC10-DET Category Distribution (Imbalanced)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Defect Category', fontsize=12, labelpad=10)
ax.set_ylabel('Number of Images', fontsize=12, labelpad=10)

# 7. 调整 Y 轴范围
ax.set_ylim(0, 1100) # 最高是 949，设到 1100 比较好看

# 8. 美化刻度和网格
plt.xticks(fontsize=10, rotation=30, ha='right') # 类别名较长，旋转 30 度防重叠
plt.yticks(fontsize=10)

# 添加 Y 轴网格线
ax.grid(axis='y', linestyle='-', alpha=0.3, color='gray')
ax.set_axisbelow(True)

# 去除边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 9. 显示图表
plt.tight_layout()
plt.show()