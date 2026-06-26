import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer


class FeatureModulationAdapter(nn.Module):
    """
    Feature Modulation Adapter (FMA)

    This module refines underwater visual tokens before vision-language fusion.
    It enhances local discriminative residuals and suppresses background-like
    responses through lightweight feature modulation.
    """

    def __init__(self, hidden_dim):
        super().__init__()

        self.local_pools = nn.ModuleList([
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=3),
        ])
        self.scale_logits = nn.Parameter(torch.zeros(3))

        self.edge_dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1,
            groups=hidden_dim, bias=False
        )

        gate_hidden = max(hidden_dim // 4, 32)
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, hidden_dim),
            nn.Sigmoid()
        )

        self.edge_scale = nn.Parameter(torch.tensor(0.10))
        self.local_scale = nn.Parameter(torch.tensor(0.50))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, visu_src):
        """
        Args:
            visu_src: visual tokens with shape [N, B, C]
        Returns:
            enhanced visual tokens with shape [N, B, C]
        """
        x = visu_src.permute(1, 2, 0)  # [B, C, N]
        b, c, n = x.shape

        hw = int(n ** 0.5)

        # Fallback for non-square token layouts.
        if hw * hw != n:
            x_seq = visu_src.permute(1, 0, 2)  # [B, N, C]
            global_mean = x_seq.mean(dim=1, keepdim=True)
            local_residual = x_seq - global_mean
            gate_in = torch.cat([local_residual, local_residual.abs()], dim=-1)
            gate = self.residual_gate(gate_in)
            enhanced_x = x_seq + self.local_scale * gate * local_residual
            return self.norm(enhanced_x).permute(1, 0, 2)

        x2d = x.view(b, c, hw, hw)

        local_residuals = []
        for pool in self.local_pools:
            local_bg = pool(x2d)
            local_residuals.append(x2d - local_bg)

        scale_weights = torch.softmax(self.scale_logits, dim=0)
        local_residual = sum(w * r for w, r in zip(scale_weights, local_residuals))

        edge_residual = self.edge_dwconv(x2d) - self.local_pools[0](x2d)

        local_tokens = local_residual.flatten(2).permute(0, 2, 1)      # [B, N, C]
        edge_tokens = edge_residual.abs().flatten(2).permute(0, 2, 1)  # [B, N, C]

        gate_in = torch.cat([local_tokens, edge_tokens], dim=-1)
        gate = self.residual_gate(gate_in)

        original_tokens = x2d.flatten(2).permute(0, 2, 1)
        enhanced_tokens = (
            original_tokens
            + self.local_scale * gate * local_tokens
            + self.edge_scale * edge_residual.flatten(2).permute(0, 2, 1)
        )

        return self.norm(enhanced_tokens).permute(1, 0, 2)


class CrossModalTokenSelector(nn.Module):
    """
    Cross-Modal Token Selector (CMTS)

    The selector softly reweights visual tokens according to their compatibility
    with the global language representation. It does not change token length,
    so it is safe for the original TransVG positional embedding and mask design.
    """

    def __init__(self, hidden_dim, keep_ratio=0.5, use_hard=False):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.use_hard = use_hard

        self.vis_proj = nn.Linear(hidden_dim, hidden_dim)
        self.txt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score_norm = nn.LayerNorm(hidden_dim)

        # Conservative residual strength for stable training.
        self.res_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, visu_src, text_global):
        """
        Args:
            visu_src: visual tokens with shape [N, B, C]
            text_global: global text feature with shape [1, B, C] or [B, C]
        Returns:
            refined visual tokens with shape [N, B, C]
        """
        if text_global.dim() == 3:
            text_global = text_global.squeeze(0)

        v = visu_src.permute(1, 0, 2)  # [B, N, C]
        b, n, c = v.shape

        v_q = F.normalize(self.vis_proj(self.score_norm(v)), dim=-1)
        t_k = F.normalize(self.txt_proj(text_global), dim=-1).unsqueeze(-1)

        score = torch.bmm(v_q, t_k).squeeze(-1)  # [B, N]

        if self.use_hard:
            k = max(1, int(n * self.keep_ratio))
            topk_idx = score.topk(k=k, dim=1).indices
            token_weight = torch.zeros_like(score)
            token_weight.scatter_(1, topk_idx, 1.0)
        else:
            token_weight = torch.softmax(score, dim=1) * n
            token_weight = token_weight.clamp(max=2.0)

        token_weight = token_weight.unsqueeze(-1)
        out = v + self.res_scale * (v * token_weight)
        return out.permute(1, 0, 2)


class TransVG(nn.Module):
    """
    VLMG implementation based on TransVG.

    Main components:
    1) multi-route text input support;
    2) MMCA-compatible visual encoder call through global text guidance;
    3) feature modulation adapter for underwater visual refinement;
    4) cross-modal token selection before VL transformer fusion.
    """

    def __init__(self, args):
        super(TransVG, self).__init__()

        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32

        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len

        self.visumodel = build_detr(args)
        self.textmodel = build_bert(args)

        num_total = self.num_visu_token + self.num_text_token + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        # self.feature_modulation = FeatureModulationAdapter(hidden_dim)
        # self.token_selector = CrossModalTokenSelector(hidden_dim, keep_ratio=0.5, use_hard=False)

        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]

        # ---------------------------------------------------------
        # 1. Encode single-route or multi-route referring expressions
        # ---------------------------------------------------------
        text_tensors = text_data.tensors
        text_masks = text_data.mask

        if text_tensors.dim() == 2:
            # Original TransVG style: [B, L]
            num_branches = 1
            L = text_tensors.shape[1]
            text_tensors_flat = text_tensors
            text_masks_flat = text_masks
        elif text_tensors.dim() == 3:
            # Multi-route text augmentation style: [B, R, L]
            _, num_branches, L = text_tensors.shape
            text_tensors_flat = text_tensors.reshape(bs * num_branches, L)
            text_masks_flat = text_masks.reshape(bs * num_branches, L)
        else:
            raise ValueError(
                "text_data.tensors should have shape [B, L] or [B, R, L], "
                f"but got {tuple(text_tensors.shape)}"
            )

        text_data_flat = type(text_data)(text_tensors_flat, text_masks_flat)

        text_fea = self.textmodel(text_data_flat)
        text_src, text_mask = text_fea.decompose()
        assert text_mask is not None

        text_src = self.text_proj(text_src)  # [B*R, L, C]
        word_feat_embed = self.mean_pooling(text_src, ~text_mask)  # [B*R, C]

        # ---------------------------------------------------------
        # Debug buffers for route-wise heatmap visualization.
        # These attributes are not used for training. They are only
        # read by external visualization scripts after forward().
        # ---------------------------------------------------------
        self.debug_num_branches = num_branches
        self.debug_text_token_len = L
        self.debug_text_mask = text_mask.detach()
        self.debug_text_tokens = text_src.detach()  # [B*R, L, C]
        self.debug_branch_text_feat = word_feat_embed.view(bs, num_branches, -1).detach()  # [B, R, C]

        # Aggregate multi-route language features for visual-side conditioning.
        word_feat_embed_fused = word_feat_embed.view(bs, num_branches, -1).mean(dim=1)
        self.debug_fused_text_feat = word_feat_embed_fused.detach()  # [B, C]
        word_feat_embed_fused = word_feat_embed_fused.unsqueeze(0)  # [1, B, C]

        # ---------------------------------------------------------
        # 2. Visual encoding with language-conditioned adaptation
        # ---------------------------------------------------------
        visu_mask, visu_src = self.visumodel(img_data, word_feat_embed_fused)
        visu_src = self.visu_proj(visu_src)  # [N, B, C]
        self.debug_visu_feat_raw = visu_src.detach()  # [N, B, C]

        # Underwater-oriented feature refinement.

        self.debug_visu_feat_modulated = visu_src.detach()  # [N, B, C]

        # Cross-modal token selection before Transformer fusion.

        self.debug_visu_feat = visu_src.detach()  # [N, B, C]

        # ---------------------------------------------------------
        # 3. Vision-language sequence construction
        # ---------------------------------------------------------
        text_src = text_src.view(bs, num_branches * L, -1).permute(1, 0, 2)
        text_mask = text_mask.view(bs, num_branches * L)

        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_mask = torch.zeros((bs, 1), device=tgt_src.device, dtype=torch.bool)

        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)

        # ---------------------------------------------------------
        # 4. Positional embeddings
        # ---------------------------------------------------------
        pos_weight = self.vl_pos_embed.weight
        tgt_pos = pos_weight[0:1]

        # Reuse text positional embeddings for multiple augmented branches.
        text_pos = pos_weight[1: 1 + L]
        text_pos_parallel = text_pos.repeat(num_branches, 1)

        visu_pos = pos_weight[1 + L:]
        custom_vl_pos = torch.cat([tgt_pos, text_pos_parallel, visu_pos], dim=0)
        vl_pos = custom_vl_pos.unsqueeze(1).repeat(1, bs, 1)

        # ---------------------------------------------------------
        # 5. VL fusion and box regression
        # ---------------------------------------------------------
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)
        vg_hs = vg_hs[0]

        pred_box = self.bbox_embed(vg_hs).sigmoid()
        return pred_box

    def mean_pooling(self, word_feat, word_mask):
        input_mask_expanded = word_mask.unsqueeze(-1).expand(word_feat.size()).float()
        sum_embeddings = torch.sum(word_feat * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def max_pooling(self, word_feat, word_mask):
        input_mask_expanded = word_mask.unsqueeze(-1).expand(word_feat.size()).float()
        embeddings = word_feat.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


