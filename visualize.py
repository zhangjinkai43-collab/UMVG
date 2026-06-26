import argparse
import textwrap
import datetime
import json
import random
import time
import math

import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import utils.misc as utils
from models_mmca_vector_based import build_model
import cv2
from datasets import build_dataset
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=0., type=float)
    parser.add_argument('--lr_visu_cnn', default=0., type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='rmsprop', type=str)
    parser.add_argument('--lr_scheduler', default='poly', type=str)
    parser.add_argument('--lr_drop', default=80, type=int)

    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")

    # Model parameters
    parser.add_argument('--model_name', type=str, default='TransVG',
                        help="Name of model to be exploited.")

    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)

    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=0, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--imsize', default=640, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')
    # Vision-Language Transformer
    parser.add_argument('--use_vl_type_embed', action='store_true',
                        help="If true, use vl_type embedding")
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--max_query_len', default=128, type=int,
                        help='maximum time steps (lang length) per batch')

    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_model', default='./saved_models/detr-r50.pth', type=str, help='detr model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # evalutaion options
    parser.add_argument('--eval_set', default='val', type=str)
    parser.add_argument('--eval_model', default='', type=str)

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_model(args)
    import torch.nn as nn
    class DummyTokenSelector(nn.Module):
        def forward(self, v, t): return v

    if hasattr(model, 'token_selector'):
        model.token_selector = DummyTokenSelector()
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # build dataset
    dataset_test = build_dataset(args.eval_set, args)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    # dataset_test  = build_dataset('test', args)

    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_test = torch.utils.data.BatchSampler(
        sampler_test, args.batch_size, drop_last=False)

    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(args.eval_model, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    # output log
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "eval_log.txt").open("a") as f:
            f.write(str(args) + "\n")
    # ================= 魔改的可视化代码 开始 =================
    print("�� 开始生成可视化结果...")
    vis_output_dir = output_dir / "vis_results"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    model_without_ddp.eval()

    # 提前准备好 ImageNet 的均值和方差，用于图像逆归一化还原
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    with torch.no_grad():
        for i, batch in enumerate(data_loader_test):
            # if i >= 10:  # 如果你只想跑前 10 个 Batch 看看效果，可以取消这行注释
            #     break

            img_data, text_data, target = batch
            img_data = img_data.to(device)
            # text_data = text_data.to(device) # 因为不需要推理预测，文本提特征也不需要了
            target = target.to(device)

            # 遍历当前 Batch 里的每一张图
            batch_size = img_data.tensors.size(0)
            for j in range(batch_size):
                # 1. 取出图像并进行逆归一化 (还原真实色彩)
                img_tensor = img_data.tensors[j]
                img_unnorm = img_tensor * std + mean
                img_unnorm = torch.clamp(img_unnorm, 0, 1)

                # 转换成 OpenCV 用的 BGR 格式 NumPy 数组
                img_cv2 = img_unnorm.permute(1, 2, 0).cpu().numpy() * 255
                img_cv2 = img_cv2.astype(np.uint8).copy()
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
                h, w, _ = img_cv2.shape

                # 2. 仅取出真实框 GT (去掉了关于 Pred 和 IoU 的所有计算)
                g_box = target[j].cpu().numpy()

                # 坐标转换函数: [cx, cy, bw, bh] -> [x1, y1, x2, y2]
                def to_xyxy(box, img_w, img_h):
                    cx, cy, bw, bh = box
                    x1 = max(0, int((cx - bw / 2) * img_w))
                    y1 = max(0, int((cy - bh / 2) * img_h))
                    x2 = min(img_w, int((cx + bw / 2) * img_w))
                    y2 = min(img_h, int((cy + bh / 2) * img_h))
                    return x1, y1, x2, y2

                gx1, gy1, gx2, gy2 = to_xyxy(g_box, w, h)

                # 3. 在图上只画 GT 绿框
                cv2.rectangle(img_cv2, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                # 加上 GT 标签，让论文图看起来更专业


                # ================= 增加：文字白板与折行写入逻辑 =================
                # 4. 获取原始长文本描述
                global_idx = i * args.batch_size + j
                raw_item = data_loader_test.dataset.images[global_idx]
                if isinstance(raw_item, dict):
                    sentence = raw_item.get('sentence', '')
                else:
                    sentence = raw_item[3]  # 兼容老数据集的元组格式

                # 5. 制作文字画板 (在图片下方外扩一块白色区域)
                max_chars_per_line = max(40, w // 10)
                wrapped_text = textwrap.wrap(sentence, width=max_chars_per_line)

                # 计算需要的额外高度
                text_board_height = len(wrapped_text) * 25 + 30

                # 创建一个全新的白色画布
                final_img = np.ones((h + text_board_height, w, 3), dtype=np.uint8) * 255

                # 把画好框的彩色原图“贴”在白色画布的上半部分
                final_img[:h, :w] = img_cv2

                # 6. 在下方的白色区域逐行写字
                y_text = h + 25
                for line in wrapped_text:
                    cv2.putText(final_img, line, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    y_text += 25
                # =============================================================

                # 7. 统一保存图片 (不再分什么好坏文件夹)
                save_path = vis_output_dir / f"dataset_sample_batch{i}_img{j}.jpg"
                cv2.imwrite(str(save_path), final_img)

        print(f"✅ 纯净版 GT 可视化完成！图片已保存至: {vis_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransVG evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)