# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils


def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, max_norm: float = 0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        img_data, text_data, target = batch

        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        # 正常接收预测框
        pred_boxes = model(img_data, text_data)

        # 获取基础定位损失 (L1 + GIoU)
        loss_dict = loss_utils.trans_vg_loss(pred_boxes, target)

        # ================= �� 创新点 1：长文本动态惩罚放大 (TLAP) =================
        max_q = args.max_query_len
        orig_mask = text_data.mask[:, :max_q]
        valid_lengths = (~orig_mask).sum(dim=1).float()

        # 计算 alpha 系数
        alpha = 1.0 + torch.clamp((valid_lengths - 10.0) / 20.0, min=0.0, max=0.5)
        mean_alpha = alpha.mean()

        # 放大损失
        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k] * mean_alpha
        # =========================================================================

        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)

        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        pred_boxes = model(img_data, text_data)
        miou, accu = eval_utils.trans_vg_eval_val(pred_boxes, target)

        accu_07 = (miou >= 0.7).float().mean() * 100.0

        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)
        metric_logger.update_v2('accu_07', accu_07, batch_size)

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats

@torch.no_grad()
def evaluate(args, model, data_loader, device):
    model.eval()

    from thop import profile

    batch = next(iter(data_loader))
    img_data, text_data, target = batch

    img_data = img_data.to(device)
    text_data = text_data.to(device)

    flops, params = profile(
        model,
        inputs=(img_data, text_data),
        verbose=False
        )

    print("GFLOPs: %.2f" % (flops / 1e9))
    print("Params(M): %.2f" % (params / 1e6))

    return 0

    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target = batch
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        output = model(img_data, text_data)
        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())


    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]

    def cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    pred_xyxy = cxcywh_to_xyxy(pred_boxes)
    gt_xyxy = cxcywh_to_xyxy(gt_boxes)

    inter_xmin = torch.max(pred_xyxy[:, 0], gt_xyxy[:, 0])
    inter_ymin = torch.max(pred_xyxy[:, 1], gt_xyxy[:, 1])
    inter_xmax = torch.min(pred_xyxy[:, 2], gt_xyxy[:, 2])
    inter_ymax = torch.min(pred_xyxy[:, 3], gt_xyxy[:, 3])

    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    gt_area = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
    union_area = pred_area + gt_area - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    accu_num_05 = (iou >= 0.5).sum().item()
    accu_num_07 = (iou >= 0.7).sum().item()

    result_tensor = torch.tensor([accu_num_05, accu_num_07, total_num]).to(device)

    torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result_tensor)

    accuracy_05 = float(result_tensor[0]) / float(result_tensor[2])
    accuracy_07 = float(result_tensor[1]) / float(result_tensor[2])

    if utils.is_main_process():
        print(f"\n" + "=" * 40)
        print(f"�� [终极评测结果] 样本总数: {int(result_tensor[2])}")
        print(f"   -> 基础定位 (Acc@0.5): {accuracy_05 * 100:.2f}%")
        print(f"   -> 高精定位 (Acc@0.7): {accuracy_07 * 100:.2f}%")
        print("=" * 40 + "\n")

    return accuracy_05