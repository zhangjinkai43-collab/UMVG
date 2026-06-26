import torch
import torch.nn.functional as F
from torch import nn
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone import build_backbone
from .transformer import build_transformer


class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_queries, train_backbone, train_transformer, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.backbone = backbone

        if self.transformer is not None:
            hidden_dim = transformer.d_model
            # ================= 新增：只保留 C4 和 C5 的投影层 =================
            self.proj_c4 = nn.Conv2d(1024, hidden_dim, kernel_size=1)
            self.proj_c5 = nn.Conv2d(2048, hidden_dim, kernel_size=1)
            # ================================================================
        else:
            hidden_dim = backbone.num_channels

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        if self.transformer is not None and not train_transformer:
            for m in [self.transformer, self.proj_c4, self.proj_c5]:
                for p in m.parameters():
                    p.requires_grad_(False)

        self.num_channels = hidden_dim

    def forward(self, samples: NestedTensor, word_feat_embed):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # features 是一个包含 [C2, C3, C4, C5] 的列表
        features, pos = self.backbone(samples, word_feat_embed)

        # ================= Lite版多尺度融合核心逻辑 =================
        # 我们果断丢弃 C2 和 C3 这两个“噪音源”，只取 C4 和 C5
        c4, mask4 = features[2].decompose()
        c5, mask5 = features[3].decompose()

        p4 = self.proj_c4(c4)
        p5 = self.proj_c5(c5)

        # 将 P4 下采样浓缩到 P5 的尺寸
        target_size = p5.shape[-2:]
        p4_down = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)

        # 降权融合：C5 是主心骨（语义），C4 只是边缘辅助（乘上 0.5 削弱噪音）
        fused_src = p5 + 0.5 * p4_down
        # ==========================================================

        mask = mask5
        pos_embed = pos[-1]

        if self.transformer is not None:
            out = self.transformer(fused_src, mask, pos_embed, word_feat_embed, query_embed=None)
        else:
            out = [mask.flatten(1), fused_src.flatten(2).permute(2, 0, 1)]

        return out


def build_detr(args):
    backbone = build_backbone(args)
    train_backbone = args.lr_visu_cnn > 0
    train_transformer = args.lr_visu_tra > 0
    if args.detr_enc_num > 0:
        transformer = build_transformer(args)
    else:
        transformer = None

    model = DETR(
        backbone, transformer, num_queries=args.num_queries,
        train_backbone=train_backbone, train_transformer=train_transformer
    )
    return model