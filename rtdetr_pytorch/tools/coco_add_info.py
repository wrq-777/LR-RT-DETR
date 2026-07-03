# tools/coco_add_info.py
import json, argparse
from pathlib import Path
from datetime import datetime

def patch_one(p):
    p = Path(p)
    j = json.loads(p.read_text('utf-8'))
    if 'info' not in j:
        j['info'] = {
            "description": "GC10 converted to COCO",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().isoformat()
        }
    if 'licenses' not in j:
        j['licenses'] = []
    # 可选：确保必要键存在
    for k in ['images','annotations','categories']:
        if k not in j:
            j[k] = []
    p.write_text(json.dumps(j, ensure_ascii=False), encoding='utf-8')
    print(f"[OK] patched: {p}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="coco json files to patch")
    args = ap.parse_args()
    for f in args.files:
        patch_one(f)
