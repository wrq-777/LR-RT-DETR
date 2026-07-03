import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. 准备你的对比数据 (请务必核对你论文 Table 1 的真实数据)
# ---------------------------------------------------------
# 模型名称
models = ['Faster R-CNN', 'SSD', 'YOLOv5s', 'YOLOv7', 'YOLOv8s', 'RT-DETR (Base)', 'GD-RT-DETR (Ours)']

# 对应的 mAP (%) - 纵坐标
mAP = [65.4, 66.8, 71.2, 72.5, 73.8, 73.7, 76.2]

# 对应的参数量 (Params, 单位: M) - 横坐标
params = [41.5, 26.2, 7.2, 36.9, 11.2, 32.3, 11.5]

# ---------------------------------------------------------
# 2. 设置绘图风格 (学术级排版)
# ---------------------------------------------------------
plt.style.use('default')
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'Times New Roman'   # 顶刊标配字体
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

# ---------------------------------------------------------
# 3. 绘制散点 (为每个模型赋予独特的形状和颜色)
# ---------------------------------------------------------
# 普通模型用灰色调，Base 用橘色，Ours 用高亮红色
colors = ['#8C92AC', '#8C92AC', '#8C92AC', '#8C92AC', '#8C92AC', '#F29F05', '#D93636']
# 分配不同形状：圆、下三角、上三角、左三角、右三角、正方形(Base)、大菱形(Ours)
markers = ['o', 'v', '^', '<', '>', 's', 'D']
sizes = [150, 150, 150, 150, 150, 200, 300]

for i in range(len(models)):
    ax.scatter(params[i], mAP[i], c=colors[i], marker=markers[i], s=sizes[i],
               edgecolors='black', linewidth=1.2, zorder=3, label=models[i])

# ---------------------------------------------------------
# 4. 添加图内标签文本 (依然保留，方便读者直观对应)
# ---------------------------------------------------------
offsets = [
    (1, -0.4),   # Faster R-CNN
    (1, -0.4),   # SSD
    (-0.5, 0.5), # YOLOv5s
    (1, -0.4),   # YOLOv7
    (-2.5, 0.5), # YOLOv8s
    (1, -0.4),   # RT-DETR (Base)
    (-2.5, 0.5)  # GD-RT-DETR
]

for i in range(len(models)):
    weight = 'bold' if 'Ours' in models[i] else 'normal'
    ax.text(params[i] + offsets[i][0], mAP[i] + offsets[i][1], models[i],
            fontsize=11, fontweight=weight, ha='left' if offsets[i][0] > 0 else 'right', zorder=4)

# ---------------------------------------------------------
# 5. 修饰坐标轴
# ---------------------------------------------------------
ax.set_xlabel('Number of Parameters (M)', fontweight='bold')
ax.set_ylabel('mAP (%) on NEU-DET', fontweight='bold')
ax.set_title('Performance Trade-off Comparison', fontweight='bold', pad=15)

# 动态调整坐标轴范围，右下角给图例留出空间
ax.set_xlim(min(params) - 5, max(params) + 15)
ax.set_ylim(min(mAP) - 2, max(mAP) + 3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------------------------------------------------------
# 6. ★★★ 绘制标准学术图例 ★★★
# ---------------------------------------------------------
# 将图例放在右下角 (lower right)
# 将图例放在绘图区外侧（右侧中间）
legend = ax.legend(title="Models", loc='center left', bbox_to_anchor=(1, 0.5),
                   frameon=True, edgecolor='black', fancybox=False, labelspacing=0.8)
legend.get_title().set_fontweight('bold')

# 【修复版本报错的地方】使用 legend_handles 替代 legendHandles
for handle in legend.legend_handles:
    handle.set_sizes([100])

# ---------------------------------------------------------
# 7. 保存与显示
# ---------------------------------------------------------
plt.tight_layout()
plt.savefig('performance_tradeoff_clean.png', dpi=300, bbox_inches='tight')
plt.savefig('performance_tradeoff_clean.pdf', bbox_inches='tight')
print("干净清晰的权衡图已生成！")
plt.show()