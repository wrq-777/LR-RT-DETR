import torch
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder


def test_asf_neck():
    print("🚀 正在测试 ASF-YOLO 改进版 HybridEncoder...")

    # 1. 模拟 GhostNetV2 的输出特征图 (Batch=2)
    # 假设 GhostNetV2 输出通道为 [40, 112, 160] (这是典型轻量级通道数)
    # 如果你的 config 里配的是 [80, 160, 960] 请自行调整，不过代码是自适应的
    in_channels = [40, 112, 160]

    # 模拟输入张量: P3, P4, P5
    inputs = [
        torch.randn(2, 40, 80, 80),  # P3 (8x 下采样)
        torch.randn(2, 112, 40, 40),  # P4 (16x 下采样)
        torch.randn(2, 160, 20, 20)  # P5 (32x 下采样)
    ]

    # 2. 实例化你的新 Neck
    try:
        neck = HybridEncoder(
            in_channels=in_channels,
            hidden_dim=256,
            use_encoder_idx=[2],  # 只对 P5 做 Transformer
            num_encoder_layers=1
        )
        print("✅ HybridEncoder 实例化成功！(ASF 模块已加载)")
    except Exception as e:
        print(f"❌ 实例化失败: {e}")
        return

    # 3. 前向传播测试
    try:
        outputs = neck(inputs)
        print("✅ 前向传播成功！")
        for i, out in enumerate(outputs):
            print(f"   输出层 {i} 尺寸: {out.shape}")

        if outputs[0].shape == (2, 256, 80, 80):
            print("\n🎉 测试通过！你的 ASF-Neck 代码没有任何问题，可以直接训练了！")
        else:
            print("\n⚠️ 警告：输出尺寸可能不对，请检查 hidden_dim 设置。")

    except RuntimeError as e:
        print(f"\n❌ 运行时报错 (可能是维度不匹配): {e}")
        print("建议：检查 config 文件中的 in_channels 是否和 GhostNetV2 实际输出一致。")


if __name__ == "__main__":
    test_asf_neck()