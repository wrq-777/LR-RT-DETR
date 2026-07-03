import torch
import cv2
import os
import sys
import numpy as np
from PIL import Image
import torchvision.transforms as T

# 获取当前脚本所在目录的上一级目录 (rtdetr_pytorch)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.core import YAMLConfig


def load_model(config_path, resume_path):
    cfg = YAMLConfig(config_path, resume=resume_path)
    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        cfg.model.load_state_dict(state)

    class Model(torch.nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            post_out = self.postprocessor(outputs, orig_target_sizes)

            # ==========================================
            # ★ 核心修复：安全解析后处理结果，兼容多种版本
            # ==========================================
            if isinstance(post_out, tuple) and len(post_out) == 3:
                # 官方常见格式: (labels, boxes, scores)
                labels, boxes, scores = post_out
                return {
                    'labels': labels[0].cpu().numpy(),
                    'boxes': boxes[0].cpu().numpy(),
                    'scores': scores[0].cpu().numpy()
                }
            elif isinstance(post_out, list) and isinstance(post_out[0], dict):
                # torchvision 标准格式: [{'labels': ..., 'boxes': ...}]
                res = post_out[0]
                return {
                    'labels': res['labels'].cpu().numpy(),
                    'boxes': res['boxes'].cpu().numpy(),
                    'scores': res['scores'].cpu().numpy()
                }
            else:
                return post_out

    model = Model().cuda()
    model.eval()
    return model


def draw_boxes(image_path, results, class_names, color=(0, 0, 255)):
    """在图片上画框 (带有边缘防越界保护)"""
    img = cv2.imread(image_path)

    if not isinstance(results, dict) or 'scores' not in results:
        return img

    labels, boxes, scores = results['labels'], results['boxes'], results['scores']

    for label, box, score in zip(labels, boxes, scores):
        if score < 0.45:
            continue

        x1, y1, x2, y2 = map(int, box)

        label = int(label)
        cls_name = class_names[label] if label < len(class_names) else f"cls_{label}"
        text = f"{cls_name} {score:.2f}"

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 计算文字需要的宽高
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # ==========================================
        # ★ 核心修复：防越界保护逻辑
        # ==========================================
        if y1 < th + 5:
            # 如果框太靠上，把文字标签画在框的【内部/下方】
            rect_y1 = y1
            rect_y2 = y1 + th + 10
            text_y = y1 + th + 5
        else:
            # 正常情况，画在框的【上方】
            rect_y1 = y1 - th - 5
            rect_y2 = y1
            text_y = y1 - 5

        # 画文字底色和文字
        cv2.rectangle(img, (x1, int(rect_y1)), (x1 + tw, int(rect_y2)), color, -1)
        cv2.putText(img, text, (x1, int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img

def main():
    # 1. 路径设置
    img_dir = "E:/datasets/NEU-DET/images/train/"
    images = ["crazing_20.jpg", "inclusion_22.jpg", "patches_22.jpg","pitted_surface_25.jpg","rolled-in_scale_66.jpg", "scratches_22.jpg"]

    # 配置相对路径 (脚本现在与配置文件夹同级)
    baseline_cfg = "configs/rtdetr/neu_det_r18.yml"
    baseline_weight = "output/neu_det_r18/best.pth"

    ours_cfg = "configs/rtdetr/neu_det_ghostv2_lgag.yml"
    ours_weight = "output/neu_det_ghostv2_lgag/best_ema.pth"

    # 修改后的代码 (1-indexed，完美对齐模型输出)
    class_names = ['background', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

    # 2. 加载模型
    print("Loading Baseline model...")
    model_base = load_model(baseline_cfg, baseline_weight)

    print("Loading GD-RT-DETR model...")
    model_ours = load_model(ours_cfg, ours_weight)

    # 包含 Resize 的预处理
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    out_dir = "visual_results"
    os.makedirs(out_dir, exist_ok=True)

    # 3. 对每张图进行推理并保存
    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"警告: 找不到图片 {img_path}")
            continue

        print(f"Processing {img_name}...")
        pil_img = Image.open(img_path).convert('RGB')
        w, h = pil_img.size
        img_tensor = transforms(pil_img).unsqueeze(0).cuda()
        orig_size = torch.tensor([[w, h]]).cuda()

        # 推理 (注意这里去掉了之前导致报错的 [0])
        with torch.no_grad():
            res_base = model_base(img_tensor, orig_size)
            res_ours = model_ours(img_tensor, orig_size)

        # 画图
        img_base_drawn = draw_boxes(img_path, res_base, class_names, color=(255, 0, 0))  # 蓝框
        img_ours_drawn = draw_boxes(img_path, res_ours, class_names, color=(0, 0, 255))  # 红框

        # 水平拼接
        combined_img = np.hstack((img_base_drawn, img_ours_drawn))

        save_path = os.path.join(out_dir, f"cmp_{img_name}")
        cv2.imwrite(save_path, combined_img)
        print(f"✅ 成功! 图片已保存至: {save_path}")


if __name__ == "__main__":
    main()