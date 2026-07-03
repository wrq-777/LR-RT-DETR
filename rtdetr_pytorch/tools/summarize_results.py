def coco_ap50(gt_json, dt_json):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(dt_json)
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    try:
        # 关键一步：填充 ev.stats
        ev.summarize()
        return float(ev.stats[1])  # AP@0.5
    except Exception:
        # 兜底：从 precision 张量手算 AP50
        P = ev.eval.get('precision', None)   # [T, R, K, A, M]
        if P is None:
            return float('nan')
        ious = ev.params.iouThrs
        i = int(np.argmin(np.abs(ious - 0.5)))
        p = P[i, :, :, 0, -1]               # 取 area=all, maxDets=100
        p = p[p > -1]
        return float(np.mean(p)) if p.size else float('nan')
