from src.nn.backbone.ghostnetv2 import GhostNetV2Backbone

if __name__ == "__main__":
    # 这里的 model_name 要和你 yml 里写的一致：ghostnetv2_1.0 或 ghostnet_1.0
    m = GhostNetV2Backbone(model_name="ghostnetv2_1.0", pretrained=True, out_indices=(2,3,4))
    shapes, chans, strides = m.probe(size=(640,640))
    print("Shapes:", shapes)         # 约等于 [1,C3,80,80], [1,C4,40,40], [1,C5,20,20]
    print("feat_channels:", chans)   # ← 这三个数就是 HybridEncoder.in_channels 的真实值
    print("feat_strides:", strides)  # 应该是 [8, 16, 32]
