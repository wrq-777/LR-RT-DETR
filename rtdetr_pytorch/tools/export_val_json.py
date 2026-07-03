# tools/export_val_json.py
import argparse, os, json, torch
from src.core.yaml_config import YAMLConfig
from src.data import get_coco_api_from_dataset
from src.solver.det_engine import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)  # checkpoint*.pth 或 checkpoint.pth
    ap.add_argument("--out", required=True)      # 输出 JSON 路径
    args = ap.parse_args()

    cfg = YAMLConfig(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cfg.model.to(device).eval()

    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)

    base_ds = get_coco_api_from_dataset(cfg.val_dataloader.dataset)
    test_stats, coco_evaluator = evaluate(
        model, cfg.criterion, cfg.postprocessor, cfg.val_dataloader, base_ds, device, output_dir=None
    )
    coco_dt = coco_evaluator.coco_eval["bbox"].cocoDt
    anns = coco_dt.dataset.get("annotations", [])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(anns, f)
    print(f"Saved COCO results JSON -> {args.out}")

if __name__ == "__main__":
    main()
