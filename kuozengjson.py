import json
import re
import os
from tqdm import tqdm


def get_spatial_phrase(bbox):
    """根据 BBox 的坐标，计算目标在图像中的绝对方位（九宫格分布）"""
    if not bbox or len(bbox) != 4:
        return "in the image"

    x, y, w, h = bbox

    # 兼容处理：判断是否为归一化坐标 (0~1)。如果不是，按默认 640x640 图片估算比例
    if w > 1 or h > 1:
        img_w, img_h = 640.0, 640.0
        # 大多数 JSON 里是 [x_min, y_min, w, h]
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
    else:
        # 如果 JSON 已经是 [cx, cy, bw, bh] 的归一化格式
        cx, cy = x, y

    # X 轴方位划分
    if cx < 0.33:
        pos_x = "left"
    elif cx > 0.66:
        pos_x = "right"
    else:
        pos_x = "middle"

    # Y 轴方位划分
    if cy < 0.33:
        pos_y = "top"
    elif cy > 0.66:
        pos_y = "bottom"
    else:
        pos_y = "center"

    # 组合空间描述
    if pos_x == "middle" and pos_y == "center":
        return "in the center of the image"
    elif pos_x == "middle":
        return f"at the {pos_y} of the image"
    elif pos_y == "center":
        return f"on the {pos_x} side of the image"
    else:
        return f"at the {pos_y} {pos_x} of the image"


def generate_four_branches(original_text, bbox):
    text = original_text.lower().strip()

    # ==========================================
    # Branch 1: Original (原句)
    # ==========================================
    txt_orig = text

    # ==========================================
    # Branch 2: Spatial (在原句基础上 + 图像绝对物理位置)
    # ==========================================
    spatial_phrase = get_spatial_phrase(bbox)
    txt_spatial = f"{txt_orig}, located {spatial_phrase}"

    # ==========================================
    # Branch 3: Semantic (核心语义提纯)
    # ==========================================
    stop_words = [
        'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'is', 'are', 'of', 'to',
        'it', 'that', 'this', 'located', 'positioned', 'which', 'there', 'some', 'has', 'have'
    ]
    words = text.replace(',', '').replace('.', '').split()
    semantic_words = [w for w in words if w not in stop_words]
    txt_semantic = " ".join(semantic_words)

    # ==========================================
    # Branch 4: Minimal (颜色 + 核心物种)
    # ==========================================
    color_lib = [
        'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange',
        'purple', 'brown', 'grey', 'gray', 'pink', 'dark', 'light', 'vibrant'
    ]
    noun_lib = [
        'sea urchin', 'starfish', 'fish', 'coral', 'rock', 'stone', 'shell',
        'crab', 'seaweed', 'plant', 'diver', 'sand', 'creature', 'animal', 'urchin'
    ]

    # 提取句子中出现的所有颜色
    found_colors = [c for c in color_lib if re.search(r'\b' + c + r'\b', text)]
    # 提取核心物种名词
    found_nouns = [n for n in noun_lib if n in text]

    # 如果没匹配到库里的名词，默认取语义提纯后的最后一个词
    noun_part = found_nouns[0] if found_nouns else (semantic_words[-1] if semantic_words else "object")
    color_part = " ".join(found_colors)

    # 拼接：颜色 + 物种
    txt_minimal = f"{color_part} {noun_part}".strip()

    return {
        "text_orig": txt_orig,
        "text_spatial": txt_spatial,
        "text_semantic": txt_semantic,
        "text_minimal": txt_minimal
    }


def augment_dataset(input_json, output_json):
    print(f"�� 正在加载数据集: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    augmented_data = []

    print("�� 开始执行四路文本精细化扩增 (Color+Species & Absolute BBox Spatial)...")
    for item in tqdm(data):
        # 兼容提取 text 和 bbox
        if isinstance(item, dict):
            sentence = item.get('sentence', '')
            if not sentence and 'sentences' in item:
                sentence = item['sentences'][0]['raw']
            bbox = item.get('bbox', [0, 0, 0, 0])
        else:
            sentence = item[3]
            bbox = item[1]

        # 核心：传入 sentence 和真实框 bbox
        four_branches = generate_four_branches(sentence, bbox)

        if isinstance(item, dict):
            item['four_branches'] = four_branches
        else:
            item = {
                'img_id': item[0],
                'bbox': item[1],
                'bbox_id': item[2],
                'sentence': item[3],
                'file_name': item[0] + '.jpg',
                'four_branches': four_branches
            }

        augmented_data.append(item)

    # ==========================================
    # 打印扩增效果展示 (以第一条数据为例)
    # ==========================================
    print("\n" + "=" * 50)
    print("✨ 全新扩增效果展示 (Sample):")
    sample_branches = augmented_data[0]['four_branches']
    print(f"  [1. 原句]     {sample_branches['text_orig']}")
    print(f"  [2. 空间]     {sample_branches['text_spatial']}")
    print(f"  [3. 语义]     {sample_branches['text_semantic']}")
    print(f"  [4. 极简]     {sample_branches['text_minimal']}")
    print("=" * 50 + "\n")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)

    print(f"�� 四路精细扩增版 JSON 已保存至: {output_json}")


if __name__ == '__main__':
    # 替换成你的 train.json 路径
    INPUT_JSON = 'val.json'
    OUTPUT_JSON = 'val_4branch.json'

    augment_dataset(INPUT_JSON, OUTPUT_JSON)