# tools/convert_gc10_to_coco.py
import os, json, random, shutil, argparse, xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# 固定 GC10 的 10 类（稳定 id 顺序）
GC10 = [
    '1_chongkong','2_hanfeng','3_yueyawan','4_shuiban','5_youban',
    '6_siban','7_yiwu','8_yahen','9_zhehen','10_yaozhe'
]
# 别名映射 / 清洗（可继续加）
ALIASES = {
    '10_yaozhed': '10_yaozhe',
    'yaozhed': '10_yaozhe',
    'yaozhe': '10_yaozhe',
    'd': None,               # 丢弃
}

def norm_name(name: str):
    n = name.strip().lower()
    n = n.replace('-', '_')
    if n in ALIASES:
        n = ALIASES[n]
    # 只接受白名单
    return n if n in GC10 else None

def build_image_index(root):
    idx = {}
    for dp, _, fns in os.walk(root):
        if os.path.basename(dp).lower() == 'lable':
            continue
        for fn in fns:
            if fn.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                idx[fn] = os.path.join(dp, fn)
    return idx

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    fname = root.findtext('filename')
    w = int(float(root.find('size/width').text))
    h = int(float(root.find('size/height').text))
    objs = []
    for obj in root.findall('object'):
        raw = obj.findtext('name') or ''
        name = norm_name(raw)
        if not name:
            print(f"[WARN] 丢弃异常类别: {raw}  @ {xml_path}")
            continue
        bb = obj.find('bndbox')
        xmin = int(float(bb.findtext('xmin'))); ymin = int(float(bb.findtext('ymin')))
        xmax = int(float(bb.findtext('xmax'))); ymax = int(float(bb.findtext('ymax')))
        xmin = max(0, min(xmin, w-1)); ymin = max(0, min(ymin, h-1))
        xmax = max(0, min(xmax, w-1)); ymax = max(0, min(ymax, h-1))
        if xmax > xmin and ymax > ymin:
            objs.append(dict(name=name, bbox=[xmin, ymin, xmax, ymax], w=w, h=h))
    return fname, w, h, objs

def coco_pack(images, ann_list, cat2id):
    coco = dict(images=[], annotations=[], categories=[])
    for cname in GC10:
        coco['categories'].append(dict(id=cat2id[cname], name=cname, supercategory='defect'))
    ann_id = 1
    for img in images:
        coco['images'].append(dict(id=img['id'], file_name=img['file_name'],
                                   width=img['width'], height=img['height']))
        for a in ann_list[img['file_name']]:
            xmin, ymin, xmax, ymax = a['bbox']
            w = xmax - xmin; h = ymax - ymin
            coco['annotations'].append(dict(
                id=ann_id, image_id=img['id'], category_id=cat2id[a['name']],
                bbox=[xmin, ymin, w, h], area=float(w*h), iscrowd=0, segmentation=[]
            ))
            ann_id += 1
    return coco

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='原始 GC10 目录，如 E:\\DATASETGC10')
    ap.add_argument('--dst', required=True, help='输出 COCO 目录，如 E:\\DATASETGC10_COCO')
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=2025)
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    img_dst_train = dst/'images'/'train'; img_dst_val = dst/'images'/'val'
    ann_dst = dst/'annotations'
    for p in [img_dst_train, img_dst_val, ann_dst]: p.mkdir(parents=True, exist_ok=True)

    img_index = build_image_index(str(src))
    xml_dir = src/'lable'
    xml_files = sorted([p for p in xml_dir.glob('*.xml')])

    per_image_anns = defaultdict(list)
    per_image_size = {}
    kept = 0; dropped = 0

    for xp in xml_files:
        fname, w, h, objs = parse_xml(str(xp))
        if fname not in img_index:
            print(f"[WARN] 找不到图片文件: {fname} (from {xp.name})"); continue
        per_image_size[fname] = (w, h)
        for o in objs:
            per_image_anns[fname].append(o); kept += 1

    # 固定 id 映射（与 GC10 顺序一致）
    cat2id = {c:i+1 for i,c in enumerate(GC10)}
    (ann_dst/'classes.json').write_text(json.dumps(dict(classes=GC10, cat2id=cat2id), ensure_ascii=False, indent=2), 'utf-8')
    print("[OK] 规范化后类别：", GC10)

    all_files = list(per_image_anns.keys())
    random.Random(args.seed).shuffle(all_files)
    n_val = int(len(all_files)*args.val_ratio)
    val_set = set(all_files[:n_val]); train_set = set(all_files[n_val:])

    def build_split(split_set, img_outdir):
        images = []; ann_list = defaultdict(list); img_id = 1
        for fname in split_set:
            src_path = img_index[fname]; dst_path = img_outdir/fname
            shutil.copy2(src_path, dst_path)
            w,h = per_image_size[fname]
            images.append(dict(id=img_id, file_name=fname, width=w, height=h))
            for a in per_image_anns[fname]:
                ann_list[fname].append(a)
            img_id += 1
        return images, ann_list

    images_tr, anns_tr = build_split(train_set, img_dst_train)
    images_va, anns_va = build_split(val_set, img_dst_val)

    coco_tr = coco_pack(images_tr, anns_tr, cat2id)
    coco_va = coco_pack(images_va, anns_va, cat2id)
    (ann_dst/'instances_train.json').write_text(json.dumps(coco_tr), 'utf-8')
    (ann_dst/'instances_val.json').write_text(json.dumps(coco_va), 'utf-8')

    print(f"[DONE] train: {len(images_tr)} val: {len(images_va)}")
    print("[TO PASTE INTO YAML] class_names =", GC10)
    print("[OUT]", str(dst))

if __name__ == '__main__':
    main()
