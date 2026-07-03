import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 1. 提取所有 4 个维度的数据！
models = ['YOLOv5', 'YOLOv7-tiny', 'YOLOv8', 'YOLOv9s', 'RT-DETR', 'LRT-DETR', 'GD-RT-DETR\n(Ours)']
params = [7.03, 6.02, 11.1, 7.2, 19.88, 6.8, 7.23]
fps = [80, 82, 88, 78, 81, 124, 116]
maps = [70.1, 68.4, 72.3, 72.9, 73.7, 74.8, 76.2]
gflops = [15.8, 13.1, 28.4, 26.7, 57.0, 13.7, 11.35]  # 新增的第四维度数据

# 2. 创建稍微宽一点的高清画布，为右侧的色度条留出空间
fig = plt.figure(figsize=(11, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#F8F9FA')

min_z = 66

# ★★★ 3. 核心：定义 GFLOPs 的颜色映射 (Red = 低算力/好，Blue = 高算力/差) ★★★
cmap = plt.get_cmap('coolwarm_r')
norm = mcolors.Normalize(vmin=10, vmax=60)  # 对应 10G 到 60G 的范围

# 4. 循环绘制每个模型
for i in range(len(models)):
    x, y, z, g = params[i], fps[i], maps[i], gflops[i]

    # 动态获取当前模型的专属颜色
    point_color = cmap(norm(g))

    # 为您的模型保留五角星形状
    if 'Ours' in models[i]:
        m = '*'
        s = 700
        fontweight = 'bold'
        text_color = '#C00000'
    else:
        m = 'o'
        s = 180
        fontweight = 'normal'
        text_color = 'black'

    # 绘制下坠线与底部阴影
    ax.plot([x, x], [y, y], [min_z, z], color='gray', linestyle='--', alpha=0.6, zorder=1)
    ax.scatter(x, y, min_z, c='gray', marker='o', s=s * 0.4, alpha=0.2, zorder=0)

    # 绘制 3D 实体点 (颜色由 GFLOPs 决定)
    ax.scatter(x, y, z, color=point_color, marker=m, s=s, edgecolor='black', linewidth=1.2, alpha=0.95, zorder=10)

    # ★★★ 修改标签：同时显示 mAP 和 GFLOPs ★★★
    label_text = f"{models[i]}\n({z}%, {g}G)"

    # 调整文字位置防止遮挡
    z_offset = 0.6
    if models[i] == 'LRT-DETR' or models[i] == 'RT-DETR':
        z_offset = -1.8
        label_text = f"{models[i]}\n({z}%, {g}G)"

    ax.text(x, y, z + z_offset, label_text, fontsize=9, fontweight=fontweight, color=text_color, ha='center',
            va='bottom', zorder=15)

# 5. 设置 3D 坐标轴
ax.set_xlabel('\nParameters (M)', fontsize=12, fontweight='bold', labelpad=15)
ax.set_ylabel('\nInference Speed (FPS)', fontsize=12, fontweight='bold', labelpad=15)
ax.set_zlabel('\nDetection Accuracy (mAP, %)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_zlim(min_z, 78)

# ★★★ 6. 在图表右侧添加 Colorbar 说明 ★★★
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
cbar.set_label('Computational Complexity (GFLOPs)\n← Lower is Better (Red)', fontsize=11, fontweight='bold',
               labelpad=10)

# 7. 调整最佳 3D 视角
ax.view_init(elev=22, azim=132)

plt.tight_layout()
plt.savefig('fig5_4D_pareto.png', dpi=300, bbox_inches='tight')
print("包含 GFLOPs 的 4D 帕累托图已生成完毕！请查看 fig5_4D_pareto.png")