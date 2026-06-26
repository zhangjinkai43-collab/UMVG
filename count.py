import json
import os


# ================= 核心统计函数 (你刚才大概率不小心把它删了) =================
def compute_dataset_stats(json_paths, dataset_name):
    """
    传入某个数据集包含的所有 json 文件路径列表 (例如 train.json 和 val.json)
    """
    unique_images = set()
    total_expressions = 0
    total_words = 0

    for path in json_paths:
        if not os.path.exists(path):
            print(f"⚠️ 找不到文件: {path}")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 兼容不同的 JSON 格式，通常是一个 list，里面装了 dict
        for item in data:
            # 1. 抓取图片 ID (根据你的 JSON 键名可能叫 img_id, image, 或者 file_name)
            img_id = item.get('img_id', item.get('image_id', item.get('file_name', None)))
            if img_id is not None:
                unique_images.add(img_id)

            # 2. 抓取文本句子 (可能叫 sentence, caption, 或者 raw_item[3])
            sentence = item.get('sentence', item.get('caption', ''))
            # 兼容元组格式
            if not sentence and isinstance(item, list) and len(item) > 3:
                sentence = item[3]

            if sentence:
                total_expressions += 1
                # 用空格切分统计单词数量
                words = sentence.split()
                total_words += len(words)

    avg_words = total_words / total_expressions if total_expressions > 0 else 0

    print(f"�� 统计完成: {dataset_name}")
    print(f"  -> 独立图片数 (Images): {len(unique_images)}")
    print(f"  -> 指代文本数 (Ref. Expressions): {total_expressions}")
    print(f"  -> 平均单词数 (Avg. Words): {avg_words:.2f}")
    print("-" * 40)


# ================= 运行调用部分 =================
if __name__ == '__main__':
    print("�� 开始读取 JSON 文件，请稍候...")

    # 统计 AquaOV255-VG
    aqua_files = [
        '/media/f517/新加卷/zjk_workspace/MMCA-main/MMCA-main/ln_data/Aquaov255/train.json',
        '/media/f517/新加卷/zjk_workspace/MMCA-main/MMCA-main/ln_data/Aquaov255/val.json'
    ]
    compute_dataset_stats(aqua_files, "AquaOV255-VG")

    # 统计 NautData-VG
    naut_files = [
        '/media/f517/新加卷/zjk_workspace/MMCA-main/MMCA-main/ln_data/nautdata/train.json',
        '/media/f517/新加卷/zjk_workspace/MMCA-main/MMCA-main/ln_data/nautdata/val.json'
    ]
    compute_dataset_stats(naut_files, "NautData-VG")