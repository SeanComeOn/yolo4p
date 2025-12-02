import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_coco_json(json_file, save_dir):
    """
    将 COCO JSON 转换为 YOLO txt 格式
    """
    # 1. 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 正在加载 {json_file} (这可能需要几秒钟)...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 构建类别映射 (COCO ID -> YOLO Index 0-79)
    # COCO 只有 80 类，但 ID 到了 90 (有些 ID 是空的)
    # 按照官方顺序重新映射为 0-79
    coco_id_to_yolo_idx = {}
    sorted_categories = sorted(data['categories'], key=lambda x: x['id'])
    for i, cat in enumerate(sorted_categories):
        coco_id_to_yolo_idx[cat['id']] = i
    
    print(f"✅ 类别映射构建完成 (共 {len(coco_id_to_yolo_idx)} 类)")

    # 3. 构建图片索引 (Image ID -> Image Info)
    # 方便通过 image_id 快速找到图片宽高和文件名
    images_info = {}
    for img in data['images']:
        images_info[img['id']] = img

    print(f"✅ 图片索引构建完成 (共 {len(images_info)} 张图)")

    # 4. 遍历所有标注并分组
    # 我们需要把属于同一张图的标注聚合在一起
    img_annotations = {} # {img_id: [ann1, ann2, ...]}
    for ann in tqdm(data['annotations'], desc="处理标注"):
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    # 5. 生成 txt 文件
    print(f"💾 开始写入 txt 文件到 {save_dir} ...")
    for img_id, info in tqdm(images_info.items(), desc="生成文件"):
        file_name = info['file_name'] # e.g., '000000123456.jpg'
        txt_name = os.path.splitext(file_name)[0] + '.txt'
        txt_path = save_path / txt_name
        
        img_w = info['width']
        img_h = info['height']
        
        # 获取该图的所有标注
        anns = img_annotations.get(img_id, [])
        
        lines = []
        for ann in anns:
            # 过滤掉 crowd (人群) 标注，通常不用于检测训练
            if ann.get('iscrowd', 0):
                continue
                
            # 获取类别索引
            cls_id = coco_id_to_yolo_idx.get(ann['category_id'])
            if cls_id is None:
                continue

            # COCO bbox: [x_min, y_min, width, height]
            box = ann['bbox']
            x_min, y_min, w, h = box[0], box[1], box[2], box[3]

            # 坐标转换 -> YOLO xywh (归一化中心点 + 宽高)
            x_center = (x_min + w / 2) / img_w
            y_center = (y_min + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            
            # 限制在 0-1 之间 (防止标注越界)
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            # 格式: class x_center y_center w h
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # 写入文件 (即使没有标注也要创建一个空文件，保持对齐)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines))

    print(f"🎉 转换完成！保存至: {save_dir}\n")

if __name__ == '__main__':
    # 配置路径 (请根据你实际解压的位置修改)
    
    # 假设你的目录结构是:
    # yolo4p/
    # ├── coco/
    # │   ├── annotations/
    # │   │   ├── instances_train2017.json
    # │   │   └── instances_val2017.json
    # │   ├── images/
    # │   └── labels/ (脚本会自动创建这个)
    
    root_dir = Path('coco2017') 
    ann_dir = root_dir / 'annotations' # 你的 json 所在目录
    
    # 1. 转换训练集
    train_json = ann_dir / 'instances_train2017.json'
    train_output = root_dir / 'labels/train2017'
    
    if train_json.exists():
        convert_coco_json(train_json, train_output)
    else:
        print(f"⚠️ 未找到 {train_json}，请检查路径。")

    # 2. 转换验证集
    val_json = ann_dir / 'instances_val2017.json'
    val_output = root_dir / 'labels/val2017'
    
    if val_json.exists():
        convert_coco_json(val_json, val_output)
    else:
        print(f"⚠️ 未找到 {val_json}，请检查路径。")