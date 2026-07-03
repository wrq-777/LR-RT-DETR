import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 将工程根目录加入系统路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig


# ==========================================
# 1. 包装模型，适配 Grad-CAM 输出
# ==========================================
class RTDETRWrapper(nn.Module):
    """
    为了让 Grad-CAM 能够反向传播，我们需要将模型的输出转化为一个标量 (Scalar)。
    目标检测中，通常使用所有预测框的最大置信度之和，或者特定类别的置信度之和作为目标。
    """

    def __init__(self, cfg_path, weight_path, device='cuda'):
        super().__init__()
        self.device = device

        # 按照 infer.py 的逻辑加载模型
        cfg = YAMLConfig(cfg_path, resume=weight_path)
        checkpoint = torch.load(weight_path, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        cfg.model.load_state_dict(state)

        # 这里不要用 deploy()，因为 deploy 会合并分类和回归头，且可能阻断梯度
        self.model = cfg.model.to(self.device)
        self.model.eval()

    def forward(self, images):
        # RT-DETR 的 forward 输出通常是一个字典，包含 'pred_logits' 和 'pred_boxes'
        outputs = self.model(images)
        # 提取分类置信度 (B, Num_Queries, Num_Classes)
        logits = outputs['pred_logits']
        # 为了可视化，我们将所有正类的预测得分相加作为梯度回传的目标
        # 注意：这里我们避开了 NMS 和后处理，直接利用 Transformer Decoder 的输出
        scores = logits.sigmoid()
        # 取每个 Query 最可能类别的得分，然后求和
        max_scores, _ = scores.max(dim=-1)
        # 返回总得分，Grad-CAM 会对这个标量求导
        return max_scores.sum(dim=-1)


# ==========================================
# 2. 主函数
# ==========================================
def main(args):
    device = torch.device(args.device)

    print("正在加载模型和权重...")
    wrapper_model = RTDETRWrapper(args.config, args.resume, device)

    # ==========================================
    # 3. 精准定位 LGAG 目标层 (Target Layer)
    # ==========================================
    # 根据你提供的 emcad_ops.py 和 hybrid_encoder_lgag.py
    # LGAG 位于 model.encoder.lgag_modules 下。
    # LGAG 的输入 x (Low-level) 经过 self.W_x (一个 Sequential，包含 Conv2d 和 BN)
    # 我们抓取 W_x 的第一层 Conv2d 作为热力图的目标层。

    # 抓取第一个 LGAG 模块 (处理最低层的高分辨率特征，细节最丰富)
    target_layers = [wrapper_model.model.encoder.lgag_modules[-1].W_x[0]]
    # 如果效果不好，也可以尝试替换为:
    # target_layers = [wrapper_model.model.encoder.lgag_modules[0].W_x[0]]

    # ==========================================
    # 4. 初始化 Grad-CAM
    # ==========================================
    # 注意：我们这里不需要 targets，因为我们在 wrapper 里已经输出标量了
    cam = GradCAM(model=wrapper_model, target_layers=target_layers, use_cuda=(args.device == 'cuda'))

    # 读取图片并预处理
    im_pil = Image.open(args.im_file).convert('RGB')
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    input_tensor = transforms(im_pil)[None].to(device)
    input_tensor.requires_grad = True  # 必须开启梯度

    # 准备用于叠加的原图 (0~1 之间)
    rgb_img = cv2.imread(args.im_file, 1)[:, :, ::1]
    rgb_img = cv2.resize(rgb_img, (640, 640))
    rgb_img = np.float32(rgb_img) / 255

    print("正在生成 LGAG 热力图...")
    # 生成 Heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # 叠加并保存
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    save_path = f"heatmap_lgag_{os.path.basename(args.im_file)}"
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"热力图已成功保存为: {save_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/rtdetr/gc10_det_ghostv2_lgag.yml')
    parser.add_argument('-r', '--resume', type=str, default='output/gc10_det_ghostv2_lgag/best.pth')
    parser.add_argument('-f', '--im-file', type=str, required=True, help='测试图片路径')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)