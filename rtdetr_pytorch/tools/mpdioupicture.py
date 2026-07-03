import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体，防止乱码（如果您在 Windows 上运行，通常默认支持；如果报错可以注释掉这行）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建 1x2 的画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ==========================================
# 左图：传统损失函数 (如 CIoU) 的痛点
# ==========================================
ax1.set_title("传统损失函数 (如 CIoU)\n痛点：中心对齐，但角点跑偏", fontsize=16, pad=20, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_aspect('equal')
ax1.axis('off') # 隐藏坐标轴，更像 PPT 插图

# 1. 绘制真实框 (Ground Truth) - 模拟细长划痕
gt_rect1 = patches.Rectangle((2, 4), 6, 1.5, linewidth=2.5, edgecolor='#2ca02c', facecolor='none', linestyle='solid', label='真实框 (Ground Truth)')
ax1.add_patch(gt_rect1)
ax1.plot(5, 4.75, marker='o', color='#2ca02c', markersize=8) # GT 中心点

# 2. 绘制预测框 (Prediction) - 中心对准了，但长宽有误差，导致框歪了
pred_rect1 = patches.Rectangle((2.5, 3.5), 5, 2.5, linewidth=2.5, edgecolor='#d62728', facecolor='none', linestyle='dashed', label='预测框 (Prediction)')
ax1.add_patch(pred_rect1)
ax1.plot(5, 4.75, marker='x', color='#d62728', markersize=10, markeredgewidth=2) # 预测框中心点与 GT 重合

ax1.legend(loc='lower center', fontsize=12, bbox_to_anchor=(0.5, -0.15))

# ==========================================
# 右图：MPDIoU 的破局之道 ("两颗钉子")
# ==========================================
ax2.set_title("MPDIoU 损失函数\n优势：强制约束左上角和右下角距离", fontsize=16, pad=20, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_aspect('equal')
ax2.axis('off')

# 1. 绘制真实框 (同左图)
gt_rect2 = patches.Rectangle((2, 4), 6, 1.5, linewidth=2.5, edgecolor='#2ca02c', facecolor='none', linestyle='solid')
ax2.add_patch(gt_rect2)
# 突出显示 GT 的左上角和右下角
ax2.plot(2, 5.5, marker='o', color='#2ca02c', markersize=10) # 左上
ax2.plot(8, 4, marker='o', color='#2ca02c', markersize=10)   # 右下

# 2. 绘制预测框
pred_rect2 = patches.Rectangle((1.2, 3.2), 7.6, 2.8, linewidth=2.5, edgecolor='#1f77b4', facecolor='none', linestyle='dashed', label='预测框 (Prediction)')
ax2.add_patch(pred_rect2)
# 突出显示 Pred 的左上角和右下角
ax2.plot(1.2, 6.0, marker='s', color='#1f77b4', markersize=8) # 左上
ax2.plot(8.8, 3.2, marker='s', color='#1f77b4', markersize=8) # 右下

# 3. 绘制 MPDIoU 核心逻辑：最小化角点距离的连线 (即“两颗钉子”)
# 左上角连线 d1
ax2.plot([2, 1.2], [5.5, 6.0], color='#ff7f0e', linestyle=':', linewidth=4)
ax2.text(1.3, 5.2, '$d_1^2$', color='#ff7f0e', fontsize=18, fontweight='bold')

# 右下角连线 d2
ax2.plot([8, 8.8], [4, 3.2], color='#ff7f0e', linestyle=':', linewidth=4)
ax2.text(8.4, 3.8, '$d_2^2$', color='#ff7f0e', fontsize=18, fontweight='bold')

# 添加解释文本框
ax2.text(5, 1.5, "目标：直接最小化 $d_1^2$ 与 $d_2^2$", ha='center', fontsize=14, color='black',
         bbox=dict(facecolor='#fff3e0', edgecolor='#ff7f0e', boxstyle='round,pad=0.6', linewidth=2))

ax2.legend(loc='lower center', fontsize=12, bbox_to_anchor=(0.5, -0.15))

# 调整布局并显示
plt.tight_layout()
plt.show()